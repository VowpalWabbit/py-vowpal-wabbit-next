API Reference
-------------

.. automodule:: vowpal_wabbit_next
   :members:
   :undoc-members:
   :member-order: bysource
   :exclude-members: PredictionType

   .. autoclass:: PredictionType

      .. autoattribute:: Scalar
         :annotation: = float

      .. autoattribute:: Scalars
         :annotation: = List[float]

      .. autoattribute:: ActionScores
         :annotation: = List[Tuple[int, float]]

      .. autoattribute:: Pdf
         :annotation: = List[Tuple[float, float, float]]

      .. autoattribute:: ActionProbs
         :annotation: = List[Tuple[int, float]]

      .. autoattribute:: Multiclass
         :annotation: = int

      .. autoattribute:: Multilabels
         :annotation: = List[int]

      .. autoattribute:: Prob
         :annotation: = float

      .. autoattribute:: DecisionProbs
         :annotation: = List[List[Tuple[int, float]]]

      .. autoattribute:: ActionPdfValue
         :annotation: = Tuple[float, float]

      .. autoattribute:: ActiveMulticlass
         :annotation: = Tuple[float, List[int]]

      .. autoattribute:: NoPred
         :annotation: = None
