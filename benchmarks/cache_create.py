import vowpal_wabbit_next as vw

workspace = vw.Workspace(["--quiet"])
with open("my_cache.cache", "wb") as cache_output_file:
    with vw.CacheFormatWriter(workspace, cache_output_file) as writer:
        with open("rcv1.10k.txt", "r") as text_input_file:
            with vw.TextFormatReader(workspace, text_input_file) as reader:
                for example in reader:
                    writer.write_example(example)
