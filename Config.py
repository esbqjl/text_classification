class Config():
    def __init__(self,target_modules=["query", "key","value"],
                 label_size=6,lora_r=8,lora_alpha=8, learning_rate = 5e-5,vocab_name ="vocab.json", **kwargs):
        self.label_size = label_size
        self.lora_r=lora_r
        self.lora_alpha = lora_alpha
        self.target_modules= target_modules
        self.learning_rate = learning_rate
        self.vocab_name = vocab_name
        for k,v in kwargs.items():
            setattr(self, k, v)
    def add(self, config_dict):
        for key, value in config_dict.items():
            setattr(self, key, value)