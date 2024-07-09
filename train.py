import os

import torch
from torch.utils.data import DataLoader
from tqdm import tqdm
from transformers import (AdamW, AutoModelForCausalLM, AutoProcessor,
                          get_scheduler)

from data import DocVQADataset

# Set device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

model = AutoModelForCausalLM.from_pretrained("./model", trust_remote_code=True, variant="by.microsoft").to(device)
# sum([p.numel() for p in model.parameters()]) 822693888
# sum([p.numel() for p in model.parameters() if p.requires_grad ]) 822693888
processor = AutoProcessor.from_pretrained("./model", trust_remote_code=True)

# [('<DocVQA>What is the chain corporate name?', 'RaceTrac.', <PIL.Image.Image image mode=RGB size=1692x2245 at 0x14D5A00FCA50>), ('<DocVQA>What is the Job Number mentioned in the document?', '244.', <PIL.Image.Image image mode=RGB size=1692x2245 at 0x14D5A0101450>)]
def collate_fn(batch):
    questions, answers, images, questionIds = zip(*batch)
    inputs = processor(
        text=list(questions), images=list(images), return_tensors="pt", padding=True
    ).to(device)
    return inputs, answers, questionIds

# Create datasets
train_dataset = DocVQADataset("train") # len(train_dataset) 39463
val_dataset = DocVQADataset("validation") # len(val_dataset) 5349

# Create DataLoader
num_workers = 0

train_loader = DataLoader(
    train_dataset,
    batch_size=4,
    collate_fn=collate_fn,
    num_workers=num_workers,
    shuffle=True,
)
val_loader = DataLoader(val_dataset, batch_size=2, collate_fn=collate_fn, num_workers=num_workers)


def train_model(train_loader, val_loader, model, processor, epochs=10, lr=1e-6):
    optimizer = AdamW(model.parameters(), lr=lr)
    num_training_steps = epochs * len(train_loader)
    lr_scheduler = get_scheduler(
        name="linear",
        optimizer=optimizer,
        num_warmup_steps=0,
        num_training_steps=num_training_steps,
    )

    for epoch in range(epochs):
        # Training phase
        model.train()
        train_loss = 0
        i = -1
        for batch in tqdm(train_loader, desc=f"Training Epoch {epoch + 1}/{epochs}"):
            i += 1
            inputs, answers, questionIds = batch
            input_ids = inputs["input_ids"]
            pixel_values = inputs["pixel_values"]
            labels = processor.tokenizer(
                text=answers,
                return_tensors="pt",
                padding=True,
                return_token_type_ids=False,
            ).input_ids.to(device)

            outputs = model(
                input_ids=input_ids, pixel_values=pixel_values, labels=labels
            )
            loss = outputs.loss

            if i % 100 == 0:
                print(loss)

                input_questions_texts = processor.batch_decode(
                    input_ids, skip_special_tokens=False
                )
                generated_ids = model.generate(
                    input_ids=inputs["input_ids"],
                    pixel_values=inputs["pixel_values"],
                    max_new_tokens=1024,
                    num_beams=3,
                )
                generated_texts = processor.batch_decode(
                    generated_ids, skip_special_tokens=False
                )

                for generated_text, answer, question, questionId in zip(generated_texts, answers, input_questions_texts, questionIds):
                    parsed_answer = processor.post_process_generation(
                        generated_text,
                        task="<DocVQA>",
                        image_size=(
                            inputs["pixel_values"].shape[-2],
                            inputs["pixel_values"].shape[-1],
                        ),
                    )
                    print("\n")
                    print("\n")
                    print("-------------------evaluation example begin------------------")
                    # we print questionId here, if you want a quick look, e.g. the questionId is 38289, you can access
                    # https://huggingface.co/datasets/zhangfaen/DocumentVQA/viewer/default/train?f[questionId][min]=38289&f[questionId][imax]=38289
                    print("Question Id:", questionId)
                    print("Question:", question)
                    print("Ground Truth:", answer)
                    print("Predication:", parsed_answer["<DocVQA>"])
                    print("-------------------evaluation example end------------------")

            loss.backward()
            optimizer.step()
            lr_scheduler.step()
            optimizer.zero_grad()

            train_loss += loss.item()

        avg_train_loss = train_loss / len(train_loader)
        print(f"Average Training Loss: {avg_train_loss}")

        # Validation phase
        model.eval()
        val_loss = 0
        with torch.no_grad():
            val_steps = 0
            for batch in tqdm(
                val_loader, desc=f"Validation Epoch {epoch + 1}/{epochs}"
            ):
                inputs, answers, questionIds = batch
                val_steps += 1
                if val_steps >= 50:
                    break

                input_ids = inputs["input_ids"]
                pixel_values = inputs["pixel_values"]
                labels = processor.tokenizer(
                    text=answers,
                    return_tensors="pt",
                    padding=True,
                    return_token_type_ids=False,
                ).input_ids.to(device)

                outputs = model(
                    input_ids=input_ids, pixel_values=pixel_values, labels=labels
                )
                loss = outputs.loss
                val_loss += loss.item()

        avg_val_loss = val_loss / val_steps
        print(f"Average Validation Loss: {avg_val_loss}")

        # Save model checkpoint
        output_dir = f"./model_checkpoints_single_gpu/epoch_{epoch+1}"
        os.makedirs(output_dir, exist_ok=True)
        model.save_pretrained(output_dir)
        processor.save_pretrained(output_dir)
        torch.save(model.state_dict(), output_dir + "/pytorch_model.finetuned.by.us.bin")

train_model(train_loader, val_loader, model, processor, epochs=3)
