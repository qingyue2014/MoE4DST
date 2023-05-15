import torch.nn as nn
class EnsembleModel(nn.Module):
    """ Dummy NMTModel wrapping individual real NMTModels """

    def __init__(self, args, models):
        super(EnsembleModel, self).__init__()
        self.models_size = len(models)
        self.base_model = models[0]
        self.eos_token_id = self.base_model.tokenizer.eos_token_id
        self.pad_token_id = self.base_model.tokenizer.pad_token_id
        self.args = args
        self.device = args.device
        self.models = models

    def forward_encoder(self, input_ids, attention_mask):
        encoder_out = []
        for model in self.models:
            eout = model.model(input_ids, attention_mask, return_encoder_outputs=True)
            encoder_out.append(eout)
        return encoder_out

    def greedy_search(self, input_ids, attention_mask, weight):
        encoder_outs = self.forward_encoder(input_ids, attention_mask)
        decoder_outs = self.base_model.model.generate(encoder_outputs=encoder_outs[0],
                                                      encoder_outputs_list=encoder_outs,
                                                      attention_mask=attention_mask,
                                                      model_list=self.models,
                                                      model_weight=weight,
                                                      max_length=15,
                                                      output_scores=True,
                                                      eos_token_id=self.eos_token_id,
                                                      return_dict_in_generate=True)
        return decoder_outs.sequences