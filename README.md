# clip_llm_ocr

### set up
Run ```sh setup.sh``` to install relevent dependencies.

### run models
Run ```sh run.sh``` to run different model architectures and see outputs in terminal. You need to include one or more architectrues as command line arguments with this command.

Possible arguments:

| Argument | Architecture to be evaluated |
|----------|----------|
| clip | bare CLIP model |
| clip_llm_short | CLIP and LLM with short prompts |
| clip_llm_long | CLIP and LLM with long prompts |
| clip_ocr | CLIP and OCR |
| clip_llm_short_ocr | CILP, LLM with short prompts, and OCR |
| clip_llm_long_ocr | CLIP, LLM with long prompts, and OCR |

For example, running ```sh run.sh clip``` will run and evaluate the bare CLIP model, running ```sh run.sh clip cilp_ocr lip_llm_long_ocr``` will run and evaluate the bare CLIP model, the CLIP + OCR architecture, and the CLIP + LLM with long prompts + OCR architecture.

