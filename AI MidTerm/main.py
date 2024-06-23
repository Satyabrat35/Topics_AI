import pandas as pd
from customModel import CustomModelInference
import os

class CustomProcessor:

    def __init__(self, input_file_path):
        """
        The__init__ function sets up the essential instance variables and initializes a model (CustomModelInference) for processing,
         along with specifying the desired output file path for the processed results.
        :param input_file_path: File Path
        """
        self.input_file_path = input_file_path
        self.model = CustomModelInference()
        self.c_values = []
        self.topic_values = []
        self.contextual_values = []
        self.output_file = f"{input_file_path}_history_size.csv"

    def perform_inference(self, index_range=None, history_size=None, model_name="gpt2"):
        """
        The function processes a range of records from a CSV file, extracts topic and story information from each record,
        applies custom language model to generate processed values based on the provided inputs, and stores these processed values.
        Finally, it updates the DataFrame with the processed values and saves the updated DataFrame to a CSV file.
        :param index_range: Custom range to perform inference on particular set of records
        :param history_size: History Size
        :param model_name: Custom Model
        :return:
        """
        df = pd.read_csv(self.input_file_path)
        try:
            if index_range:
                df = df.iloc[index_range[0]: index_range[1]]
            for index, row in df.iterrows():
                print(f"Processing story row {self.output_file} {index + 1} / {len(df)}")
                topic = row['summary']
                story = row['story']
                result = self.model.process_sentences(story, topic, history_size=history_size, model=model_name)
                self.c_values.append(result[0])
                self.topic_values.append(result[1])
                self.contextual_values.append(result[2])

        except Exception as e:
            print("Exception occurred: ", e)
        finally:
            df['c_value'] = self.c_values
            df['topic_output'] = self.topic_values
            df['contextual'] = self.contextual_values

            df.to_csv(self.output_file)

if __name__ == "__main__":
    # Provide your dataset file path, history size and call the custom model
    hippo_path = "/Users/satya/Desktop/AI MidTerm/data/hippo_filtered.csv"
    history_size = 5
    CustomProcessor(hippo_path).perform_inference(history_size=history_size)
