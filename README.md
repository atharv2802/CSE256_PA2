# CSE256_PA2
## Atharv Biradar

===========================================================
                        README
===========================================================

Instructions to Run the Code
===========================================================

To execute the code, run the following command in your terminal:

    python main.py

You will be prompted to choose a part to run from the following options:

    Choose a part:
    1. Encoder-only Classification Model
    2. Decoder-only Language Modeling
    3. Transformer with AliBi and RoPE
    4. Exit
    Enter a number:

Enter the corresponding number to select the desired part:

1. **Encoder-only Classification Model**: Runs the classification model using only the encoder component of the Transformer architecture.
2. **Decoder-only Language Modeling**: Runs the language modeling task using only the decoder component of the Transformer architecture.
3. **Transformer with AliBi and RoPE**: Executes the Transformer model with advanced positional encoding techniques: AliBi and RoPE.
4. **Exit**: Exits the program.

Additional Setup Requirements
===========================================================

- **NLTK Setup**: You will need to download the NLTK punkt tokenizer for text processing. Run the following command in Python to download it:

    ```python
    import nltk
    nltk.download('punkt')
    ```

- **CPU Execution**: All tasks are executed on the CPU and not the GPU.

Additional Notes
===========================================================

- Ensure that all necessary datasets are available in the specified directory before running the tasks.
