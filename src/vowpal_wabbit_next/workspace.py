from typing import List, Optional, Tuple, Union
import typing
from vowpal_wabbit_next import _core, Example

import numpy as np
import numpy.typing as npt

PredictionType = _core.PredictionType
LabelType = _core.LabelType

ScalarPrediction = float
ScalarsPrediction = List[float]
ActionScoresOrProbsPrediction = List[Tuple[int, float]]
DecisionScoresPrediction = List[List[Tuple[int, float]]]
MulticlassPrediction = int
MultilabelsPrediction = List[int]
PdfPrediction = List[Tuple[float, float, float]]
ActionPdfValuePrediction = Tuple[float, float]
ActiveMulticlassPrediction = Tuple[int, List[int]]
NoPrediction = type(None)

Prediction = Union[
    ScalarPrediction,
    ScalarsPrediction,
    ActionScoresOrProbsPrediction,
    DecisionScoresPrediction,
    MulticlassPrediction,
    MultilabelsPrediction,
    PdfPrediction,
    ActionPdfValuePrediction,
    ActiveMulticlassPrediction,
    NoPrediction,
]


class Workspace:
    def __init__(
        self,
        args: List[str],
        *,
        model_data: Optional[bytes] = None,
        _existing_workspace: Optional[_core.Workspace] = None
    ):
        """Main object used for making predictions and training a model.

        The VW library logs various things while running. There are two streams of logging exposed, which can be accessed via the standard Python logging interface.
        * `vowpal_wabbit_next.log` - VW's structured logging stream. If it outputs a warning it will be logged here.
        * `vowpal_wabbit_next.driver` - This is essentially the CLI driver output. This is rarely needed from Python.

        See the logging example below.

        Examples:
            Load a model from a file:

            >>> from vowpal_wabbit_next import Workspace
            >>> with open("model.bin", "rb") as f:
            ...     model_data = f.read()
            >>> workspace = Workspace([], model_data=model_data)

            Create a workspace for training a contextual bandit with action dependent features model:

            >>> from vowpal_wabbit_next import Workspace
            >>> workspace = Workspace(["--cb_explore_adf"])

            Outputting structured logging messages from VW:

            >>> from vowpal_wabbit_next import Workspace
            >>> import logging
            >>> logging.basicConfig(level=logging.INFO)
            >>> logging.getLogger("vowpal_wabbit_next.log").setLevel("INFO")
            >>> workspace = Workspace([])

        Args:
            args (List[str]): VowpalWabbit command line options for configuring the model. An overall list can be found `here <https://vowpalwabbit.org/docs/vowpal_wabbit/python/latest/command_line_args.html>`_. Some options are unsupported which are:
                `--sort_features`, `--ngram`, `--feature_limit`, `--ignore`.
            model_data (Optional[bytes], optional): Bytes of a VW model to be loadeed.
            _existing_workspace (Optional[_core.Workspace], optional): This is for internal usage and should not be set by a user.
        """
        if _existing_workspace is not None:
            self._workspace = _existing_workspace
        else:
            self._workspace = _core.Workspace(args, model_data=model_data)

    def predict_one(self, example: typing.Union[Example, List[Example]]) -> Prediction:
        """Make a single prediction.

        Examples:
            >>> from vowpal_wabbit_next import Workspace, TextFormatParser
            >>> workspace = Workspace([])
            >>> parser = TextFormatParser(workspace)
            >>> workspace.learn_one(parser.parse_line("1.0 | price:.18 sqft:.15 age:.35 1976"))
            >>> workspace.predict_one(parser.parse_line("| price:.53 sqft:.32 age:.87 1924"))
            1.0

        Args:
            example (typing.Union[Example, List[Example]]): Example to use for prediction. This should be a list if this workspace is :py:meth:`vowpal_wabbit_next.Workspace.multiline`, otherwise it is should be a single Example

        Returns:
            Prediction: Prediction produced by this example. The type corresponds to the :py:meth:`vowpal_wabbit_next.Workspace.prediction_type` of the model. See :py:class:`vowpal_wabbit_next.PredictionType` for the mapping to types.
        """
        if isinstance(example, Example):
            return self._workspace.predict_one(example)
        else:
            return self._workspace.predict_multi_ex_one(example)

    def learn_one(self, example: typing.Union[Example, List[Example]]) -> None:
        """Learn from one single example. Note, passing a list of examples here means the input is a multiline example, and not several individual examples. The label type of the example must match what is returned by :py:meth:`vowpal_wabbit_next.Workspace.label_type`.

        Examples:
            >>> from vowpal_wabbit_next import Workspace, TextFormatParser
            >>> workspace = Workspace([])
            >>> parser = TextFormatParser(workspace)
            >>> workspace.learn_one(parser.parse_line("1.0 | price:.18 sqft:.15 age:.35 1976"))
            >>> workspace.predict_one(parser.parse_line("| price:.53 sqft:.32 age:.87 1924"))
            1.0

        Args:
            example (typing.Union[Example, List[Example]]): Example to learn on.
        """
        if isinstance(example, Example):
            self._workspace.learn_one(example)
        else:
            self._workspace.learn_multi_ex_one(example)

    def predict_then_learn_one(
        self, example: typing.Union[Example, List[Example]]
    ) -> Prediction:
        """Make a prediction then learn from the example. This is potentially more efficient than a predict_one call followed by a learn_one call as the implementation is able to avoid duplicated work as long as the prediction is guaranteed to be from before learning.

        Examples:
            >>> from vowpal_wabbit_next import Workspace, TextFormatParser
            >>> workspace = Workspace([])
            >>> parser = TextFormatParser(workspace)
            >>> workspace.predict_then_learn_one(parser.parse_line("1.0 | price:.18 sqft:.15 age:.35 1976"))
            0.0

        Args:
            example (typing.Union[Example, List[Example]]): Example to use for prediction. This should be a list if this workspace is :py:meth:`vowpal_wabbit_next.Workspace.multiline`, otherwise it is should be a single Example

        Returns:
            Prediction: Prediction produced by this example. The type corresponds to the :py:meth:`vowpal_wabbit_next.Workspace.prediction_type` of the model. See :py:class:`vowpal_wabbit_next.PredictionType` for the mapping to types.
        """
        if isinstance(example, Example):
            return self._workspace.predict_then_learn_one(example)
        else:
            return self._workspace.predict_then_learn_multi_ex_one(example)

    @property
    def prediction_type(self) -> PredictionType:
        """Based on the command line parameters used to setup VW a certain type of prediction is produced. See :py:class:`vowpal_wabbit_next.PredictionType` for the list of types and their corresponding Python type.

        Returns:
            PredictionType: The type of prediction this Workspace produces
        """
        return self._workspace.get_prediction_type()

    @property
    def label_type(self) -> LabelType:
        """Based on the command line parameters used to setup VW a certain label type is required.
        This can also be thought of as the type of problem being solved.

        Returns:
            LabelType: The type of label Examples must have to be used by this Workspace
        """
        return self._workspace.get_label_type()

    @property
    def multiline(self) -> bool:
        """Based on the command line parameters used to setup VW, the input to learn, predict or parsers expects either single Examples or lists of Examples.

        Returns:
            bool: True if this Workspace is configured as multiline, otherwise False
        """
        return self._workspace.get_is_multiline()

    def serialize(self) -> bytes:
        """Serialize the current workspace as a VW model that can be loaded by the Workspace constructor, or command line tool.

        Returns:
            bytes: raw bytes of serialized Workspace
        """
        return self._workspace.serialize()

    def weights(self) -> npt.NDArray[np.float32]:
        """Access to the weights of the model currently.

        This function returns a view of the weights and any changes to the returned array will be reflected in the model.

        There are 3 dimensions:

        * The feature index (aka weight index)
        * The index of the interleaved model, which should usually be 0
        * The weight itself and the extra state stored with the weight

        .. attention::
            Only dense weights are supported.

        .. warning::
            This is an experimental feature.

        Examples:
            >>> from vowpal_wabbit_next import Workspace
            >>> model = Workspace([])
            >>> print(model.weights().shape)
            (262144, 1, 4)

        Returns:
            np.ndarray: Array of weights
        """
        return np.array(self._workspace.weights(), copy=False)

    def json_weights(
        self, *, include_feature_names: bool = False, include_online_state: bool = False
    ) -> str:
        """Debugging utility which dumps the weights in the model currently as a JSON string.

        .. warning::
            This is an experimental feature.

        Args:
            include_feature_names (bool, optional): Includes the feature names and interaction terms in the output. This requires the workspace to be configured to support it. This is not well exposed to Python currently but the way to do it is:
                `--dump_json_weights_experimental=unused --dump_json_weights_include_feature_names_experimental`
            include_online_state (bool, optional): Includes extra save_resume state in the output.This requires the workspace to be configured to support it. This is not well exposed to Python currently but the way to do it is:
                `--dump_json_weights_experimental=unused --dump_json_weights_include_extra_online_state_experimental`

        Returns:
            str: JSON string representing model weights
        """
        return self._workspace.json_weights(
            include_feature_names=include_feature_names,
            include_online_state=include_online_state,
        )

    def get_index_for_scalar_feature(
        self,
        feature_name: str,
        *,
        feature_value: Optional[str] = None,
        namespace_name: str = " "
    ) -> int:
        """Calculate the has for a given feature.

        The logic is rather complicated to work out an index. This function also takes into account index
        truncation caused by the index multiplier taking the index out of the standard weight space.

        .. warning::
            This is an experimental feature, the interface may change.

        Examples:
            >>> from vowpal_wabbit_next import Workspace
            >>> model = Workspace([])
            >>> # Feature which looks like "|test thing" in text format
            >>> model.get_index_for_scalar_feature("thing", namespace_name="test")
            148099

        Args:
            feature_name (str): The name of the feature
            feature_value (Optional[str], optional): String value of feature. If passed chain hashing will be used. In text format this looks like `feature_name:feature_value`
            namespace_name (str, optional): Namespace of feature. Defaults to " " which is the default namespace.

        Returns:
            int: The index of the feature
        """
        return self._workspace.get_index_for_scalar_feature(
            feature_name=feature_name,
            feature_value=feature_value,
            namespace_name=namespace_name,
        )

    # TODO implement
    # def get_index_for_interacted_feature(
    #     self, terms: List[Tuple[str, Optional[str], str]]
    # ) -> int:
    #     """Calculate the hash for an interacted feature.

    #     .. warning::
    #         This is an experimental feature, the interface may change.

    #     Args:
    #         terms (List[Tuple[str, Optional[str], str]]): List of features which are interacted. (feature_name, feature_value, namespace_name)

    #     Returns:
    #         int: The index of the feature
    #     """
    #     raise NotImplementedError()
