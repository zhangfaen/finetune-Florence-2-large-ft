import os
from functools import partial

import friendlywords as fw
import torch
import torch.distributed as dist
import torch.multiprocessing as mp
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler
from tqdm import tqdm
from transformers import (AdamW, AutoModelForCausalLM, AutoProcessor, get_scheduler)

from data import DocVQADataset

os.environ["TOKENIZERS_PARALLELISM"] = "false"


def setup(rank, world_size):
    os.environ["MASTER_ADDR"] = "localhost"
    os.environ["MASTER_PORT"] = "12355"
    dist.init_process_group("nccl", rank=rank, world_size=world_size)
    torch.cuda.set_device(rank)


def cleanup():
    dist.destroy_process_group()


def collate_fn(batch, processor, device):
    questions, answers, images, questionIds = zip(*batch)
    inputs = processor(
        text=list(questions), images=list(images), return_tensors="pt", padding=True
    ).to(device)
    return inputs, answers, questionIds


def create_data_loaders(
    train_dataset,
    val_dataset,
    batch_size,
    num_workers,
    rank,
    world_size,
    processor,
    device,
):
    train_sampler = DistributedSampler(
        train_dataset, num_replicas=world_size, rank=rank
    )
    val_sampler = DistributedSampler(val_dataset, num_replicas=world_size, rank=rank)

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        collate_fn=partial(collate_fn, processor=processor, device=device),
        num_workers=num_workers,
        sampler=train_sampler,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        collate_fn=partial(collate_fn, processor=processor, device=device),
        num_workers=num_workers,
        sampler=val_sampler,
    )

    return train_loader, val_loader


def train_model(rank, world_size, epochs=3, lr=1e-6, eval_steps=300):
    setup(rank, world_size)
    device = torch.device(f"cuda:{rank}")
    run_name = fw.generate(2, separator="_")

    train_dataset = DocVQADataset(split = 'train')
    val_dataset = DocVQADataset(split = 'validation')
    
    # Load the model and processor
    model = AutoModelForCausalLM.from_pretrained("./model", trust_remote_code=True, variant="by.microsoft").to(device)
    processor = AutoProcessor.from_pretrained("./model", trust_remote_code=True)
    model = DDP(model, device_ids=[rank])

    # Create DataLoaders
    batch_size = 4
    num_workers = 0
    train_loader, val_loader = create_data_loaders(
        train_dataset,
        val_dataset,
        batch_size,
        num_workers,
        rank,
        world_size,
        processor,
        device,
    )

    optimizer = AdamW(model.parameters(), lr=lr)
    num_training_steps = epochs * len(train_loader)
    lr_scheduler = get_scheduler(
        name="linear",
        optimizer=optimizer,
        num_warmup_steps=0,
        num_training_steps=num_training_steps,
    )
    global_step = 0

    for epoch in range(epochs):
        # Training phase
        model.train()
        train_loss = 0
        for batch in tqdm(train_loader, desc=f"Training Epoch {epoch + 1}/{epochs}", position=rank):
            inputs, answers, questionIds = batch

            # Prepare the input and target tensors
            input_ids = inputs["input_ids"].to(device)
            pixel_values = inputs["pixel_values"].to(device)
            labels = processor.tokenizer(
                text=answers,
                return_tensors="pt",
                padding=True,
                return_token_type_ids=False,
            ).input_ids.to(device)

            outputs = model(input_ids=input_ids, pixel_values=pixel_values, labels=labels)
            loss = outputs.loss
            loss.backward()
            optimizer.step()
            lr_scheduler.step()
            optimizer.zero_grad()

            train_loss += loss.item()
            global_step += 1

            if global_step % eval_steps == 0:
                # Evaluation phase
                model.eval()
                val_loss = 0
                val_steps = 0
                with torch.no_grad():
                    for batch in tqdm(val_loader, desc=f"Evaluation at step {global_step}", position=rank):
                        inputs, answers, questionIds = batch
                        val_steps += 1
                        if val_steps >= 50:
                            break
                        # Prepare the input and target tensors
                        input_ids = inputs["input_ids"].to(device)
                        pixel_values = inputs["pixel_values"].to(device)
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
                print(f"Rank {rank} - Step {global_step} - Average Validation Loss: {avg_val_loss}")

                model.train()
                
        avg_train_loss = train_loss / len(train_loader)
        print(f"Rank {rank} - Average Training Loss: {avg_train_loss}")

        # Save model checkpoint
        if rank == 0:  # Only the main process saves the checkpoint
            output_dir = f"./model_checkpoints_ddp/{run_name}/epoch_{epoch+1}"
            os.makedirs(output_dir, exist_ok=True)
            model.module.save_pretrained(output_dir)
            processor.save_pretrained(output_dir)
            torch.save(model.module.state_dict(), output_dir + "/pytorch_model.finetuned.by.us.bin")


    cleanup()


def main():
    world_size = torch.cuda.device_count()
    mp.spawn(
        train_model,
        args=(world_size,),
        nprocs=world_size,
        join=True
    )

if __name__ == "__main__":
    main()
