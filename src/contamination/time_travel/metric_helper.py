"""
Code from: https://github.com/shahriargolchin/time-travel-in-llms/blob/main/src/helpers/metric_helper.py
"""
import evaluate


class Bleurt:
    def __init__(self, checkpoint="BLEURT-20", batch_size=16):
        self.batch_size = batch_size
        self._bleurt_scorer = self._load_bleurt(checkpoint)

    def _load_bleurt(self, checkpoint: str):
        try:
            from contamination.time_travel.bleurt_loader import BleurtLoader

            loader = BleurtLoader(checkpoint=checkpoint)
            loader.prepare_module()
            from bleurt_scorer.bleurt import score as bleurt_scorer

            return bleurt_scorer.BleurtScorer(loader.model_path)
        except ImportError:
            raise ImportError(
                "BLEURT could not be loaded. Ensure BLEURT dependencies are "
                "available if this module is needed."
            )

    def score(self, references, candidates):
        if self._bleurt_scorer is None:
            raise Exception(
                "Score calculation is unavailable as the BLEURT module "
                "could not be loaded."
            )
        return self._bleurt_scorer.score(
            references=references, candidates=candidates, batch_size=self.batch_size
        )


class Rouge:
    def __init__(self, rouge_type="rougeL"):
        self.rouge_scorer = evaluate.load("rouge")
        self.rouge_type = rouge_type

    def score(self, references, candidates):
        rouge_scores = self.rouge_scorer.compute(
            references=references, predictions=candidates, use_aggregator=False
        )
        return rouge_scores.get(self.rouge_type, "Invalid ROUGE type")
