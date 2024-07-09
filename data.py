from datasets import load_dataset
from torch.utils.data import Dataset


class BaseDataset(Dataset):
    def __init__(self, split):
        self._split = split
        self.data = []
        self.task_prompt = ""

    def __len__(self):
        return len(self.data)

    def correct_casing_finqa(self, text, is_question=False):
        if text and text[0].islower():
            text = text.capitalize()
        if not text.endswith(".") and not is_question:
            text += "."
        if not text.endswith("?") and is_question:
            text += "?"
        return text


class DocVQADataset(BaseDataset):
    def __init__(self, split):
        super().__init__(split)
        import pdb
        # pdb.set_trace()
        self.data = load_dataset("zhangfaen/DocumentVQA", split=split)
        
        self.task_prompt = "<DocVQA>"

    def __getitem__(self, idx):
        # import pdb
        # pdb.set_trace()
        example = self.data[idx]
        question = self.task_prompt + self.correct_casing_finqa(
            example["question"], True
        )
        first_answer = example["answers"][0]
        answers = self.correct_casing_finqa(first_answer)
        image = example["image"]  # The image is already a PIL Image object
        if image.mode != "RGB":
            image = image.convert("RGB")
        return question, answers, image, example['questionId']
    