import torch
import json
from tqdm import tqdm
import warnings

class EvalModel():
    def __init__(self, model=None, tokenizer=None, device="cuda:0", predictions:list=[], references:list=[], adapter_path=None):
        if predictions != [] and len(predictions) != len(references):
            warnings.warn("The length of predictions is not equal to references.")
        self.predictions = predictions
        self.references = references
        self.model = model
        self.tokenizer = tokenizer
        self.device = device
        self.result = {}

        if adapter_path is not None:
            load_lora_model = LoadLoraModel(model, adapter_path)
            self.model = load_lora_model.load_lora_adapter()
    
    def _gen_input(self, data):
        # Placeholder for actual implementation
        return None
    
    def _gen_pred_input(self, data):
        return None

    def _get_response(self, input_data, max_new_tokens):
        # Placeholder for actual implementation
        return None

    def _get_response_batch(self, input_data_batch, max_new_tokens, **kwargs):
        # Placeholder for actual implementation
        return None
    
    def _get_pred_batch(self, response, **kwargs):
        response_batch = [self._gen_pred_input(r) for r in response]
        response_batch = self._get_response_batch(response_batch, **kwargs)
        return response_batch
    
    def _gen_pre_ref(self, response, data):
        self.references.append(data['answer'])
        self.predictions.append(response)
        return None
    
    def _gen_pre_ref_batch(self, response_batch, data_batch):
        self.references.extend([data["answer"] for data in data_batch])
        self.predictions.extend(response_batch)
        return None

    def check_model_and_tokenizer(self):
        if self.model is None or self.tokenizer is None:
            print("No model or tokenizer loaded.")
            return False
        return True
    
    def inference(self, dataset, max_new_tokens=1024):
        if not self.check_model_and_tokenizer():
            return None, None
        for i in tqdm(range(len(dataset))):
            data = dataset[i]
            input_data = self._gen_input(data)
            response = self._get_response(input_data, max_new_tokens=max_new_tokens)

            self._gen_pre_ref(response, data)
        return self.predictions, self.references
    
    def inference_batch(self, dataset, max_new_tokens=1024, batch_size=8, **kwargs):
        if not self.check_model_and_tokenizer():
            return None, None
        for data_id in tqdm(range(0, len(dataset), batch_size)):
            data_batch = dataset[data_id:data_id+batch_size]
            input_data_batch = []
            for data in data_batch:
                input_data_batch.append(self._gen_input(data))
            response_batch = self._get_response_batch(input_data_batch, max_new_tokens=max_new_tokens, **kwargs)
            if not isinstance(response_batch, list):
                raise TypeError("Response batch must be a list")

            self._gen_pre_ref_batch(response_batch, data_batch)
    
    def get_eval_result(self):
        return self.result

    def load_pre_ref(self, pre_path, ref_path):
        try:
            with open(pre_path, "rb") as file:
                self.predictions = json.load(file)
            with open(ref_path, "rb") as file:
                self.references = json.load(file)
            if len(self.predictions) != len(self.references):
                warnings.warn("The length of predictions is not equal to references.")
        except Exception as e:
            print(f"Failed to load predictions and references: {e}")

    def load_model(self, model):
        self.model = model

    def load_tokenizer(self, tokenizer):
        self.tokenizer = tokenizer

    def clear_pre_ref(self):
        self.predictions = []
        self.references = []
    
    def save_pre_ref(self, save_path):
        try:
            prefix = ""
            with open(f"{save_path}/{prefix}predictions.json", "w", encoding="utf-8") as file:
                json.dump(self.predictions, file, ensure_ascii=False)
            with open(f"{save_path}/{prefix}references.json", "w", encoding="utf-8") as file:
                json.dump(self.references, file, ensure_ascii=False)
        except Exception as e:
            print(f"Failed to save predictions and references: {e}")
    
    def save_eval_result(self, result_path):
        try:
            with open(result_path, "w", encoding="utf-8") as file:
                json.dump(self.result, file, ensure_ascii=False)
        except Exception as e:
            print(f"Failed to save evaluation result: {e}")

class EvalQwen(EvalModel):
    def _get_response(self, input_data, max_new_tokens=1024):
        text = self.tokenizer.apply_chat_template(
            input_data,
            tokenize=False,
            add_generation_prompt=True
        )
        model_inputs = self.tokenizer([text], return_tensors="pt").to(self.device)

        generated_ids = self.model.generate(
            model_inputs.input_ids,
            max_new_tokens=max_new_tokens
        )
        generated_ids = [
            output_ids[len(input_ids):] for input_ids, output_ids in zip(model_inputs.input_ids, generated_ids)
        ]

        response = self.tokenizer.batch_decode(generated_ids, skip_special_tokens=True)[0]
        return response
    
    def _get_response_batch(self, input_data_batch, max_new_tokens=1024):
        text_batch = []
        for input_data in input_data_batch:
            text_batch.append(self.tokenizer.apply_chat_template(
                input_data,
                tokenize=False,
                add_generation_prompt=True,
                padding = True,
                truncation=True,
                max_length=2048
            ))
        model_inputs = self.tokenizer(text_batch, padding=True, truncation=True, max_length=2048, return_tensors="pt").to(self.device)

        generated_ids = self.model.generate(
            model_inputs.input_ids,
            max_new_tokens=max_new_tokens
        )
        generated_ids = [
            output_ids[len(input_ids):] for input_ids, output_ids in zip(model_inputs.input_ids, generated_ids)
        ]

        response = self.tokenizer.batch_decode(generated_ids, skip_special_tokens=True)

        return response