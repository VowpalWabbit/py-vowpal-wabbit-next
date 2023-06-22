API Reference
-------------

.. automodule:: vowpal_wabbit_next
   :members:
   :undoc-members:
   :member-order: bysource
   :special-members: __getitem__, __contains__, __iter__, __delitem__
   :exclude-members: PredictionType, LabelType

   .. autoclass:: PredictionType

      Enum where each variant corresponds to Python types for the different prediction types.

      .. autoattribute:: Scalar
         :annotation: = float

      .. autoattribute:: Scalars
         :annotation: = List[float]

      .. autoattribute:: ActionScores
         :annotation: = List[Tuple[int, float]]

         Where the tuple is (`action_index`, `score`) and `action_index` is zero based.

      .. autoattribute:: Pdf
         :annotation: = List[Tuple[float, float, float]]

         Where the tuple is (`left`, `right`, `value`)

      .. autoattribute:: ActionProbs
         :annotation: = List[Tuple[int, float]]

         Where the tuple is (`action_index`, `probability`) and `action_index` is zero based.

      .. autoattribute:: Multiclass
         :annotation: = int

      .. autoattribute:: Multilabels
         :annotation: = List[int]

      .. autoattribute:: Prob
         :annotation: = float

      .. autoattribute:: DecisionProbs
         :annotation: = List[List[Tuple[int, float]]]

         Where the tuple is (`action_index`, `probability`) and `action_index` is zero based.

      .. autoattribute:: ActionPdfValue
         :annotation: = Tuple[float, float]

         Where the tuple is (`action`, `value`)

      .. autoattribute:: ActiveMulticlass
         :annotation: = Tuple[int, List[int]]

         Where the tuple is (`predicted_class`, `more_info_required_for_classes`)

      .. autoattribute:: NoPred
         :annotation: = None


   .. autoclass:: LabelType

      Enum where each variant corresponds to Python types for the different label types.

      .. autoattribute:: Simple
         :annotation:

         Type: :class:`~vowpal_wabbit_next.SimpleLabel`

      .. autoattribute:: CB
         :annotation:

         Type: :class:`~vowpal_wabbit_next.CBLabel`

      .. autoattribute:: CBEval
         :annotation:

         Type: *NotImplemented*

      .. autoattribute:: CS
         :annotation:

         Type: :class:`~vowpal_wabbit_next.CSLabel`

      .. autoattribute:: CCB
         :annotation:

         Type: *NotImplemented*

      .. autoattribute:: Continuous
         :annotation:

         Type: *NotImplemented*

      .. autoattribute:: Multiclass
         :annotation:

         Type: :class:`~vowpal_wabbit_next.MulticlassLabel`

      .. autoattribute:: Multilabel
         :annotation:

         Type: *NotImplemented*

      .. autoattribute:: NoLabel
         :annotation:

         Type: ``None``

      .. autoattribute:: Slates
         :annotation:

         Type: *NotImplemented*

.. autoclass:: vowpal_wabbit_next._core.FeatureGroupRef
   :members:
   :undoc-members:
