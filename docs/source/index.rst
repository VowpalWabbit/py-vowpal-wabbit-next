vowpal_wabbit_next
==================

.. important::
   If you are looking for `VowpalWabbit's <https://github.com/VowpalWabbit/vowpal_wabbit>`_
   existing Python bindings go `here <https://pypi.org/project/vowpalwabbit/>`_

`vowpal_wabbit_next` is a new set of Python bindings for VowpalWabbit. Many
things have been changed, or rethought, to make using this API easier and safer.
For example, there is no `finish_example` or `setup_example` in this API. If you
don't know what that means then don't worry.

They are highly experimental and the API may change significantly. If you'd
prefer a more stable package then the `existing bindings
<https://pypi.org/project/vowpalwabbit/>`_ are still available.

Installation
------------

.. code-block:: bash

   pip install vowpal-wabbit-next


.. toctree::
   :caption: How-to guides
   :hidden:

   how_to_guides/dataset_readers.ipynb
   how_to_guides/save_load_models.ipynb
   how_to_guides/cache_format.ipynb
   how_to_guides/inspect_model_weights.ipynb

.. toctree::
   :caption: Tutorials
   :hidden:

   tutorials/contextual_bandit_content_personalization.ipynb

.. toctree::
   :caption: Reference
   :hidden:

   reference
   vw_versions
