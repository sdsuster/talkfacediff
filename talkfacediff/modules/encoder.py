from transformers import Wav2Vec2Processor, HubertModel
import torch
import pytorch_lightning as pl

class AudioEncoder(pl.LightningModule):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        # Load pretrained HuBERT model and processor
        self.processor = Wav2Vec2Processor.from_pretrained("facebook/hubert-base-ls960")
        self.model = HubertModel.from_pretrained("facebook/hubert-base-ls960")
        self.model.eval()

    def forward(self, audio, sr):
        input_values = self.processor(audio, sampling_rate=sr, return_tensors="pt").input_values

        # Extract features
        with torch.no_grad():
            return self.model(input_values)
        