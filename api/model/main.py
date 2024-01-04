from config import config as cfg
cfg.BATCH_SIZE = 1

from trainer import Trainer

trainer = Trainer(architecture='trans', hidden_dim=32, lr=1e-3)
trainer.load_model("l1_600e_res.pt")

print("==== EVALUATION ====")
trainer.evaluate("test")
print("======= DONE =======")

trainer.predict_to_json()

while(True):
    continue