
from transformers import GPT2Tokenizer, GPT2LMHeadModel, BartForConditionalGeneration, BartTokenizer, RobertaTokenizer, RobertaForSequenceClassification, LogitsProcessorList, AutoModelForSeq2SeqLM, AutoTokenizer

import torch

class MutliDecoderModel():


    def __init__(self,tokenizer_name, model_names):
        self.tokenizer_name = tokenizer_name
        self.tokenizer = AutoTokenizer.from_pretrained(self.tokenizer_name)
        self.model_names = model_names
        self.models = []
        for name in self.model_names:
            self.models.append(BartForConditionalGeneration.from_pretrained(name))


    def generate(self,
        input_ids: torch.LongTensor,
        beam_scorer: BeamScorer,
        logits_processor: Optional[LogitsProcessorList] = None,
        stopping_criteria: Optional[StoppingCriteriaList] = None,
        max_length: Optional[int] = None,
        pad_token_id: Optional[int] = None,
        eos_token_id: Optional[Union[int, List[int]]] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        output_scores: Optional[bool] = None,
        return_dict_in_generate: Optional[bool] = None,
        synced_gpus: bool = False,
        **model_kwargs,):

           # init values
        logits_processor = logits_processor if logits_processor is not None else LogitsProcessorList()
        stopping_criteria = stopping_criteria if stopping_criteria is not None else StoppingCriteriaList()
        if max_length is not None:
            warnings.warn(
                "`max_length` is deprecated in this function, use"
                " `stopping_criteria=StoppingCriteriaList(MaxLengthCriteria(max_length=max_length))` instead.",
                UserWarning,
            )
            stopping_criteria = validate_stopping_criteria(stopping_criteria, max_length)
        if len(stopping_criteria) == 0:
            warnings.warn("You don't have defined any stopping_criteria, this will likely loop forever", UserWarning)
        pad_token_id = pad_token_id if pad_token_id is not None else self.generation_config.pad_token_id
        eos_token_id = eos_token_id if eos_token_id is not None else self.generation_config.eos_token_id
        if isinstance(eos_token_id, int):
            eos_token_id = [eos_token_id]
        output_scores = output_scores if output_scores is not None else self.generation_config.output_scores
        output_attentions = (
            output_attentions if output_attentions is not None else self.generation_config.output_attentions
        )
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.generation_config.output_hidden_states
        )
        return_dict_in_generate = (
            return_dict_in_generate
            if return_dict_in_generate is not None
            else self.model[0].generation_config.return_dict_in_generate
        )

        batch_size = len(beam_scorer._beam_hyps)
        num_beams = beam_scorer.num_beams

        batch_beam_size, cur_len = input_ids.shape

        if num_beams * batch_size != batch_beam_size:
            raise ValueError(
                f"Batch dimension of `input_ids` should be {num_beams * batch_size}, but is {batch_beam_size}."
            )

        # init attention / hidden states / scores tuples
        scores = () if (return_dict_in_generate and output_scores) else None
        beam_indices = (
            tuple(() for _ in range(batch_beam_size)) if (return_dict_in_generate and output_scores) else None
        )
        decoder_attentions = () if (return_dict_in_generate and output_attentions) else None
        cross_attentions = () if (return_dict_in_generate and output_attentions) else None
        decoder_hidden_states = () if (return_dict_in_generate and output_hidden_states) else None

        # if model is an encoder-decoder, retrieve encoder attention weights and hidden states
        if return_dict_in_generate and self.model[0].config.is_encoder_decoder:
            encoder_attentions = model_kwargs["encoder_outputs"].get("attentions") if output_attentions else None
            encoder_hidden_states = (
                model_kwargs["encoder_outputs"].get("hidden_states") if output_hidden_states else None
            )

        # initialise score of first beam with 0 and the rest with -1e9. This makes sure that only tokens
        # of the first beam are considered to avoid sampling the exact same tokens across all beams.
        beam_scores = torch.zeros((batch_size, num_beams), dtype=torch.float, device=input_ids.device)
        beam_scores[:, 1:] = -1e9
        beam_scores = beam_scores.view((batch_size * num_beams,))

        this_peer_finished = False  # used by synced_gpus only
        while True:
            if synced_gpus:
                # Under synced_gpus the `forward` call must continue until all gpus complete their sequence.
                # The following logic allows an early break if all peers finished generating their sequence
                this_peer_finished_flag = torch.tensor(0.0 if this_peer_finished else 1.0).to(input_ids.device)
                # send 0.0 if we finished, 1.0 otherwise
                dist.all_reduce(this_peer_finished_flag, op=dist.ReduceOp.SUM)
                # did all peers finish? the reduced sum will be 0.0 then
                if this_peer_finished_flag.item() == 0.0:
                    break

            model_inputs1 = self.models[0].prepare_inputs_for_generation(input_ids, **model_kwargs)
            model_inputs2 = self.models[1].prepare_inputs_for_generation(input_ids, **model_kwargs)

            outputs1 = self.models[0](
                **model_inputs1,
                return_dict=True,
                output_attentions=output_attentions,
                output_hidden_states=output_hidden_states,
            )

            outputs2 = self.models[0](
                **model_inputs2,
                return_dict=True,
                output_attentions=output_attentions,
                output_hidden_states=output_hidden_states,
            )

            if synced_gpus and this_peer_finished:
                cur_len = cur_len + 1
                continue  # don't waste resources running the code we don't need

            next_token_logits1 = outputs1.logits[:, -1, :]
            next_token_logits2 = outputs2.logits[:, -1, :]
            # hack: adjust tokens for Marian. For Marian we have to make sure that the `pad_token_id`
            # cannot be generated both before and after the `nn.functional.log_softmax` operation.
            next_token_logits1 = self.model[0].adjust_logits_during_generation(next_token_logits1, cur_len=cur_len)
            next_token_logits2 = self.model[1].adjust_logits_during_generation(next_token_logits2, cur_len=cur_len)

            next_token_scores1 = nn.functional.log_softmax(
                next_token_logits1, dim=-1
            )  # (batch_size * num_beams, vocab_size)

            next_token_scores2 = nn.functional.log_softmax(
                next_token_logits2, dim=-1
            )  # (batch_size * num_beams, vocab_size)

            next_token_scores_processed1 = logits_processor(input_ids, next_token_scores1)
            next_token_scores1 = next_token_scores_processed1 + beam_scores[:, None].expand_as(next_token_scores1)



            next_token_scores_processed2 = logits_processor(input_ids, next_token_scores2)
            next_token_scores2 = next_token_scores_processed2 + beam_scores[:, None].expand_as(next_token_scores2)


            next_token_scores = (next_token_scores1 + next_token_scores2)/2

            next_token_scores_processed = (next_token_scores_processed1+next_token_scores_processed2)/2



            # Store scores, attentions and hidden_states when required
            # if return_dict_in_generate:
            #     if output_scores:
            #         scores += (next_token_scores_processed,)
            #     if output_attentions:
            #         decoder_attentions += (
            #             (outputs.decoder_attentions,) if self.config.is_encoder_decoder else (outputs.attentions,)
            #         )
            #         if self.config.is_encoder_decoder:
            #             cross_attentions += (outputs.cross_attentions,)

            #     if output_hidden_states:
            #         decoder_hidden_states += (
            #             (outputs.decoder_hidden_states,)
            #             if self.config.is_encoder_decoder
            #             else (outputs.hidden_states,)
            #         )

            # reshape for beam search
            vocab_size = next_token_scores.shape[-1]
            next_token_scores = next_token_scores.view(batch_size, num_beams * vocab_size)

            # Sample 2 next tokens for each beam (so we have some spare tokens and match output of beam search)
            next_token_scores, next_tokens = torch.topk(
                next_token_scores, 2 * num_beams, dim=1, largest=True, sorted=True
            )

            next_indices = torch.div(next_tokens, vocab_size, rounding_mode="floor")
            next_tokens = next_tokens % vocab_size

            # stateless
            beam_outputs = beam_scorer.process(
                input_ids,
                next_token_scores,
                next_tokens,
                next_indices,
                pad_token_id=pad_token_id,
                eos_token_id=eos_token_id,
                beam_indices=beam_indices,
            )

            beam_scores = beam_outputs["next_beam_scores"]
            beam_next_tokens = beam_outputs["next_beam_tokens"]
            beam_idx = beam_outputs["next_beam_indices"]

            input_ids = torch.cat([input_ids[beam_idx, :], beam_next_tokens.unsqueeze(-1)], dim=-1)

            model_kwargs1 = self.model[0]._update_model_kwargs_for_generation(
                outputs, model_kwargs2, is_encoder_decoder=self.config.is_encoder_decoder
            )
            if model_kwargs1["past_key_values"] is not None:
                model_kwargs1["past_key_values"] = self.model[0]._reorder_cache(model_kwargs1["past_key_values"], beam_idx)


            model_kwargs2 = self.model[1]._update_model_kwargs_for_generation(
                outputs, model_kwargs2, is_encoder_decoder=self.config.is_encoder_decoder
            )
            if model_kwargs2["past_key_values"] is not None:
                model_kwargs2["past_key_values"] = self.model[1]._reorder_cache(model_kwargs2["past_key_values"], beam_idx)

            if return_dict_in_generate and output_scores:
                beam_indices = tuple((beam_indices[beam_idx[i]] + (beam_idx[i],) for i in range(len(beam_indices))))

            # increase cur_len
            cur_len = cur_len + 1

            if beam_scorer.is_done or stopping_criteria(input_ids, scores):
                if not synced_gpus:
                    break
                else:
                    this_peer_finished = True

        sequence_outputs = beam_scorer.finalize(
            input_ids,
            beam_scores,
            next_tokens,
            next_indices,
            pad_token_id=pad_token_id,
            eos_token_id=eos_token_id,
            max_length=stopping_criteria.max_length,
            beam_indices=beam_indices,
        )

        if return_dict_in_generate:
            if not output_scores:
                sequence_outputs["sequence_scores"] = None

            if self.config.is_encoder_decoder:
                return BeamSearchEncoderDecoderOutput(
                    sequences=sequence_outputs["sequences"],
                    sequences_scores=sequence_outputs["sequence_scores"],
                    scores=scores,
                    beam_indices=sequence_outputs["beam_indices"],
                    encoder_attentions=encoder_attentions,
                    encoder_hidden_states=encoder_hidden_states,
                    decoder_attentions=decoder_attentions,
                    cross_attentions=cross_attentions,
                    decoder_hidden_states=decoder_hidden_states,
                )
            else:
                return BeamSearchDecoderOnlyOutput(
                    sequences=sequence_outputs["sequences"],
                    sequences_scores=sequence_outputs["sequence_scores"],
                    scores=scores,
                    beam_indices=sequence_outputs["beam_indices"],
                    attentions=decoder_attentions,
                    hidden_states=decoder_hidden_states,
                )
        else:
            return sequence_outputs["sequences"]