from fairseq.models.transformer.transformer_legacy import transformer_iwslt_de_en

def load_fairseq_config(config_name, args):
    if config_name == 'transformer_iwslt_de_en':
        transformer_iwslt_de_en(args)
        return
    else:
        raise ValueError(f'Not implemented: {config_name}')