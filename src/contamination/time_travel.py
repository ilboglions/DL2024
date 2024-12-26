import wandb
from tqdm import tqdm


PROMPTS = {
    "general": """INSTRUCTION:
Finish SENTENCE 2 based on SENTENCE 1, such that the following LABEL shows the logical relationship between SENTENCE 1 and SENTENCE 2.

SENTENCE 1:
{premise}

LABEL: {label}

SENTENCE 2:
""",
    "guided": """INSTRUCTION:
You are provided with SENTENCE 1 from the {split} split of the {dataset} dataset.
Finish SENTENCE 2 as appeared in the dataset.
SENTENCE 2 MUST EXACTLY match the instance in the dataset.

SENTENCE 1:
{premise}

LABEL: {label}

SENTENCE 2:
""",
}

def generate(model, tokenizer, dataset, device):
    completions = []

    for sample in tqdm(dataset):
        input_ids = sample['input_ids'].unsqueeze(0).to(device)
        print(f"Input shape:{input_ids.shape}")
        outputs = model.generate(
            input_ids,
            do_sample=False,
            max_new_tokens=50,
            eos_token_id=tokenizer.eos_token_id)
        print(f"Output shape: {outputs.shape}")
        text_outputs = tokenizer.batch_decode(outputs, skip_special_tokens=True)
        print(text_outputs)
        completions.append(text_outputs)

def eval_tt(model, tokenizer, general_datasets, guided_datasets, device):
    general_completions = {}
    guided_completions = {}

    for dataset_name, dataset in general_datasets.items():
        print(f"Generating general completions for {dataset_name} ...")
        general_completions[dataset_name] = generate(model, tokenizer, dataset, device)

    for dataset_name, dataset in guided_datasets.items():
        print(f"Generating guided completions for {dataset_name} ...")
        guided_completions[dataset_name] = generate(model, tokenizer, dataset, device)

# class Alg1EvalPhase(ExperimentResultSaver):
#     def __init__(self, df, args, scoring_tool, save_intermediate_results):
#         super().__init__(df, args.filepath, args.experiment, save_intermediate_results)
#         self.df = df
#         self.args = args
#         self.scoring_tool = scoring_tool
#         self.metric = str(scoring_tool.__class__.__name__).lower().strip()
#         self.filepath = Path(args.filepath)

#     def text_prep(self):
#         if self.args.task == "nli":
#             return TextPrep.nli_text_prep(self.df, self.args.text_column)
#         elif self.metric == "bleurt":
#             return TextPrep.blert_text_prep(self.df, self.args.text_column)
#         else:
#             return TextPrep.default_text_prep(self.df)

#     def evaluate_score(self, references, general_candidates, guided_candidates):
#         general_scores = self.scoring_tool.score(
#             references=references, candidates=general_candidates
#         )
#         guided_scores = self.scoring_tool.score(
#             references=references, candidates=guided_candidates
#         )

#         general_scores = [round(score, 2) for score in general_scores]
#         guided_scores = [round(score, 2) for score in guided_scores]

#         self.df[f"{self.metric}_score_for_general_completion"] = general_scores
#         self.df[f"{self.metric}_score_for_guided_completion"] = guided_scores

#         return general_scores, guided_scores

#     def resampling_and_save(self, general_scores, guided_scores):
#         resampling_processor = ResamplingProcessor(num_resample=10_000)

#         result_filepath = (
#             self.experiment
#             / f"{self.metric}_resampling_results_for_{self.filepath.stem}.txt"
#         )

#         resampling_processor.save_results(
#             general=general_scores,
#             guided=guided_scores,
#             metric=self.metric,
#             result_filepath=result_filepath,
#         )

#     def evaluate(self):
#         logger.info(f"Starting evaluation using {self.metric} ...")

#         if (
#             "generated_guided_completion" not in self.df.columns
#             or "generated_general_completion" not in self.df.columns
#         ):
#             raise ValueError(
#                 f"For evaluation using {self.metric}, completions from general "
#                 "and guided instructions must be provided. If you have these "
#                 "completions, make sure they are listed as 'generated_general_completion' "
#                 "and 'generated_guided_completion' in the csv file. "
#                 "Otherwise, you need to get these completions by running "
#                 "--process_general_replication and --process_guided_replication, respectively."
#             )
#         references, general_candidates, guided_candidates = self.text_prep()

#         # sanity check
#         logger.info(f"Example of reference text: {references[0]}")
#         logger.info(f"Example of general completion: {general_candidates[0]}")
#         logger.info(f"Example of guided completion: {guided_candidates[0]}")

#         general_scores, guided_scores = self.evaluate_score(
#             references=references,
#             general_candidates=general_candidates,
#             guided_candidates=guided_candidates,
#         )

#         self.save_to_csv()

#         self.resampling_and_save(
#             general_scores=general_scores, guided_scores=guided_scores
#         )

#         return self.df
