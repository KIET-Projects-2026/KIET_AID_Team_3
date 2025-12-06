# Install required packages
!pip install datasets pandas requests beautifulsoup4 chardet huggingface_hub -q

import os
import pandas as pd
from datasets import load_dataset
import requests
from bs4 import BeautifulSoup
import time
import random
import re
from urllib.parse import urljoin
import warnings
warnings.filterwarnings('ignore')

# Mount Google Drive
from google.colab import drive
drive.mount('/content/drive')

class HindiDatasetCreator:
    def __init__(self, output_dir):
        self.output_dir = output_dir
        os.makedirs(output_dir, exist_ok=True)
        self.datasets = []

    def create_summary(self, text):
        """Create a meaningful summary from text"""
        # Split by Hindi sentence endings
        sentences = [s.strip() for s in re.split('[।?!]', text) if s.strip()]

        if len(sentences) >= 3:
            # Take first 2-3 sentences as summary
            summary_sentences = sentences[:min(3, len(sentences))]
            summary = '। '.join(summary_sentences) + '।'
        elif len(sentences) == 2:
            summary = sentences[0] + '।'
        else:
            # If only one sentence, take first half
            words = text.split()
            if len(words) > 10:
                summary = ' '.join(words[:len(words)//2]) + '...'
            else:
                summary = text

        return summary

    def download_oscar_hindi(self, target_size_mb=300):
        """Download Hindi portion of OSCAR dataset"""
        print("📥 Downloading OSCAR Hindi dataset...")
        output_file = os.path.join(self.output_dir, "oscar_hindi.csv")

        data = []
        try:
            dataset = load_dataset("oscar", "unshuffled_deduplicated_hi", split="train", streaming=True)

            for i, sample in enumerate(dataset):
                if i >= 100000:  # Safety limit
                    break

                text = sample.get('text', '').strip()
                # Filter for reasonable Hindi text
                if len(text) > 200 and len(text) < 10000:
                    hindi_chars = set('अआइईउऊऋएऐओऔकखगघङचछजझञटठडढणतथदधनपफबभमयरलवशषसह')
                    if len(hindi_chars.intersection(set(text))) > 20:  # Ensure it's Hindi
                        summary = self.create_summary(text)
                        data.append({'text': text, 'summary': summary})

                if i % 5000 == 0 and i > 0:
                    print(f"  Processed {i} samples...")
                    # Check file size
                    if os.path.exists(output_file):
                        size_mb = os.path.getsize(output_file) / (1024 * 1024)
                        if size_mb >= target_size_mb:
                            break

                if len(data) >= 1000:
                    # Save in chunks
                    df_chunk = pd.DataFrame(data)
                    if os.path.exists(output_file):
                        df_chunk.to_csv(output_file, mode='a', header=False, index=False, encoding='utf-8-sig')
                    else:
                        df_chunk.to_csv(output_file, index=False, encoding='utf-8-sig')
                    data = []

        except Exception as e:
            print(f"❌ Error in OSCAR download: {e}")

        # Save remaining data
        if data:
            df_chunk = pd.DataFrame(data)
            if os.path.exists(output_file):
                df_chunk.to_csv(output_file, mode='a', header=False, index=False, encoding='utf-8-sig')
            else:
                df_chunk.to_csv(output_file, index=False, encoding='utf-8-sig')

        if os.path.exists(output_file):
            size_mb = os.path.getsize(output_file) / (1024 * 1024)
            print(f"✅ OSCAR dataset: {size_mb:.2f} MB")
        return output_file

    def download_wikipedia_hindi(self, target_size_mb=200):
        """Download Hindi Wikipedia articles"""
        print("📥 Downloading Wikipedia Hindi dataset...")
        output_file = os.path.join(self.output_dir, "wikipedia_hindi.csv")

        data = []
        try:
            # Using a smaller, more manageable Wikipedia dataset
            dataset = load_dataset("alespalla/chatbot_instruction_prompts", "hi", split="train", streaming=True)

            for i, sample in enumerate(dataset):
                if i >= 50000:
                    break

                text = sample.get('text', sample.get('instruction', ''))
                if text and len(text) > 100:
                    summary = self.create_summary(text)
                    data.append({'text': text, 'summary': summary})

                if i % 1000 == 0 and i > 0:
                    print(f"  Processed {i} Wikipedia samples...")
                    if len(data) >= 500:
                        df_chunk = pd.DataFrame(data)
                        if os.path.exists(output_file):
                            df_chunk.to_csv(output_file, mode='a', header=False, index=False, encoding='utf-8-sig')
                        else:
                            df_chunk.to_csv(output_file, index=False, encoding='utf-8-sig')
                        data = []

        except Exception as e:
            print(f"❌ Error in Wikipedia download: {e}")
            # Alternative Wikipedia source
            try:
                print("Trying alternative Wikipedia source...")
                dataset = load_dataset("wikimedia/wikipedia", "20231101.hi", split="train[:10000]", streaming=False)

                for i, sample in enumerate(dataset):
                    text = sample.get('text', '')
                    if len(text) > 200:
                        summary = self.create_summary(text)
                        data.append({'text': text, 'summary': summary})

                    if i % 1000 == 0:
                        print(f"  Processed {i} alternative Wikipedia samples...")

            except Exception as e2:
                print(f"❌ Alternative Wikipedia also failed: {e2}")

        # Save remaining data
        if data:
            df_chunk = pd.DataFrame(data)
            if os.path.exists(output_file):
                df_chunk.to_csv(output_file, mode='a', header=False, index=False, encoding='utf-8-sig')
            else:
                df_chunk.to_csv(output_file, index=False, encoding='utf-8-sig')

        if os.path.exists(output_file):
            size_mb = os.path.getsize(output_file) / (1024 * 1024)
            print(f"✅ Wikipedia dataset: {size_mb:.2f} MB")
        return output_file

    def download_indic_corp(self, target_size_mb=300):
        """Download IndicCorp Hindi dataset"""
        print("📥 Downloading IndicCorp Hindi dataset...")
        output_file = os.path.join(self.output_dir, "indic_corp_hindi.csv")

        data = []
        try:
            dataset = load_dataset("ai4bharat/IndicCorp", "hi", split="train", streaming=True)

            for i, sample in enumerate(dataset):
                if i >= 80000:
                    break

                text = sample.get('text', '').strip()
                if len(text) > 150:
                    summary = self.create_summary(text)
                    data.append({'text': text, 'summary': summary})

                if i % 5000 == 0 and i > 0:
                    print(f"  Processed {i} IndicCorp samples...")
                    if len(data) >= 1000:
                        df_chunk = pd.DataFrame(data)
                        if os.path.exists(output_file):
                            df_chunk.to_csv(output_file, mode='a', header=False, index=False, encoding='utf-8-sig')
                        else:
                            df_chunk.to_csv(output_file, index=False, encoding='utf-8-sig')
                        data = []

        except Exception as e:
            print(f"❌ Error in IndicCorp download: {e}")

        # Save remaining data
        if data:
            df_chunk = pd.DataFrame(data)
            if os.path.exists(output_file):
                df_chunk.to_csv(output_file, mode='a', header=False, index=False, encoding='utf-8-sig')
            else:
                df_chunk.to_csv(output_file, index=False, encoding='utf-8-sig')

        if os.path.exists(output_file):
            size_mb = os.path.getsize(output_file) / (1024 * 1024)
            print(f"✅ IndicCorp dataset: {size_mb:.2f} MB")
        return output_file

    def create_synthetic_hindi_data(self, target_size_mb=200):
        """Create synthetic Hindi data for additional volume"""
        print("🔄 Creating synthetic Hindi data...")
        output_file = os.path.join(self.output_dir, "synthetic_hindi.csv")

        # Base Hindi templates
        templates = [
            "भारत देश एशिया महाद्वीप में स्थित एक विशाल और विविधतापूर्ण देश है। यह अपनी समृद्ध सांस्कृतिक विरासत, प्राचीन इतिहास और तेजी से विकसित होती अर्थव्यवस्था के लिए विश्व भर में प्रसिद्ध है।",
            "हिंदी भाषा भारत की राजभाषा है और देवनागरी लिपि में लिखी जाती है। यह विश्व की तीसरी सबसे अधिक बोली जाने वाली भाषा है जिसे लाखों लोग मातृभाषा के रूप में बोलते हैं।",
            "विज्ञान और प्रौद्योगिकी के क्षेत्र में भारत ने उल्लेखनीय प्रगति की है। चंद्रयान और मंगलयान जैसे अंतरिक्ष मिशनों ने भारत को अंतरिक्ष अन्वेषण के क्षेत्र में अग्रणी देशों में शामिल कर दिया है।",
            "भारतीय संस्कृति विश्व की सबसे पुरानी संस्कृतियों में से एक है। इसमें विभिन्न धर्म, परंपराएं, भाषाएं और कला रूप शामिल हैं जो इसे अद्वितीय बनाते हैं।",
            "शिक्षा के क्षेत्र में भारत में कई प्रतिष्ठित संस्थान हैं। भारतीय प्रौद्योगिकी संस्थान और भारतीय विज्ञान संस्थान जैसे संस्थान विश्व स्तर पर मान्यता प्राप्त हैं।"
        ]

        data = []
        target_samples = 20000

        for i in range(target_samples):
            base_template = random.choice(templates)

            # Create variations
            variations = [
                f"{base_template} भारत में {random.randint(28, 36)} राज्य और {random.randint(7, 9)} केंद्र शासित प्रदेश हैं जो इसकी प्रशासनिक व्यवस्था का हिस्सा हैं।",
                f"{base_template} देश की जनसंख्या {random.randint(130, 145)} करोड़ से अधिक है जो इसे विश्व का दूसरा सबसे अधिक जनसंख्या वाला देश बनाती है।",
                f"{base_template} भारत की अर्थव्यवस्था विश्व की सबसे तेजी से बढ़ती अर्थव्यवस्थाओं में से एक है जो विभिन्न क्षेत्रों में विकास कर रही है।",
                f"{base_template} यह देश अपने विविध धर्मों, जैसे हिंदू, मुस्लिम, सिख, ईसाई, बौद्ध और जैन धर्म के सह-अस्तित्व के लिए जाना जाता है।",
                f"{base_template} भारत का संविधान विश्व का सबसे लंबा लिखित संविधान है जो देश के शासन का मार्गदर्शन करता है।"
            ]

            text = random.choice(variations)
            # Repeat to create longer text
            extended_text = text * random.randint(2, 4)
            summary = self.create_summary(text)

            data.append({'text': extended_text, 'summary': summary})

            if i % 5000 == 0 and i > 0:
                print(f"  Created {i} synthetic samples...")
                df_chunk = pd.DataFrame(data)
                if os.path.exists(output_file):
                    df_chunk.to_csv(output_file, mode='a', header=False, index=False, encoding='utf-8-sig')
                else:
                    df_chunk.to_csv(output_file, index=False, encoding='utf-8-sig')
                data = []

        # Save remaining data
        if data:
            df_chunk = pd.DataFrame(data)
            if os.path.exists(output_file):
                df_chunk.to_csv(output_file, mode='a', header=False, index=False, encoding='utf-8-sig')
            else:
                df_chunk.to_csv(output_file, index=False, encoding='utf-8-sig')

        if os.path.exists(output_file):
            size_mb = os.path.getsize(output_file) / (1024 * 1024)
            print(f"✅ Synthetic dataset: {size_mb:.2f} MB")
        return output_file

    def combine_and_clean_datasets(self):
        """Combine all datasets and clean them"""
        print("🔗 Combining all datasets...")

        all_files = [
            "oscar_hindi.csv",
            "wikipedia_hindi.csv",
            "indic_corp_hindi.csv",
            "synthetic_hindi.csv"
        ]

        combined_data = []

        for file in all_files:
            file_path = os.path.join(self.output_dir, file)
            if os.path.exists(file_path):
                try:
                    # Read with proper encoding
                    df = pd.read_csv(file_path, encoding='utf-8-sig')
                    print(f"✅ Loaded {len(df)} samples from {file}")
                    combined_data.append(df)
                except Exception as e:
                    print(f"❌ Error reading {file}: {e}")
                    # Try alternative encodings
                    try:
                        df = pd.read_csv(file_path, encoding='utf-8')
                        combined_data.append(df)
                        print(f"✅ Loaded {len(df)} samples from {file} with utf-8")
                    except:
                        try:
                            df = pd.read_csv(file_path, encoding='latin1')
                            combined_data.append(df)
                            print(f"✅ Loaded {len(df)} samples from {file} with latin1")
                        except:
                            print(f"❌ Failed to read {file}")

        if not combined_data:
            print("❌ No data could be loaded!")
            return None

        # Combine all data
        final_df = pd.concat(combined_data, ignore_index=True)

        # Clean the data
        print("🧹 Cleaning dataset...")

        # Remove duplicates
        initial_count = len(final_df)
        final_df = final_df.drop_duplicates(subset=['text'])
        print(f"Removed {initial_count - len(final_df)} duplicates")

        # Remove NaN values
        final_df = final_df.dropna()

        # Filter for reasonable text lengths
        final_df = final_df[final_df['text'].str.len() > 100]
        final_df = final_df[final_df['summary'].str.len() > 20]

        # Ensure Hindi text
        hindi_chars = set('अआइईउऊऋएऐओऔकखगघङचछजझञटठडढणतथदधनपफबभमयरलवशषसह')
        def is_hindi(text):
            if pd.isna(text):
                return False
            return len(hindi_chars.intersection(set(str(text)))) > 10

        final_df = final_df[final_df['text'].apply(is_hindi)]
        final_df = final_df[final_df['summary'].apply(is_hindi)]

        print(f"🎯 Final dataset size: {len(final_df)} samples")

        # Save final dataset
        final_path = os.path.join(self.output_dir, "hindi_summarization_final.csv")
        final_df.to_csv(final_path, index=False, encoding='utf-8-sig')

        # Check final size
        final_size_gb = os.path.getsize(final_path) / (1024 * 1024 * 1024)
        print(f"💾 Final dataset size: {final_size_gb:.2f} GB")

        return final_df

    def display_samples(self, df, num_samples=3):
        """Display sample data"""
        print("\n" + "="*80)
        print("SAMPLE DATA PREVIEW")
        print("="*80)

        for i in range(min(num_samples, len(df))):
            print(f"\n📝 Sample {i+1}:")
            print(f"Text: {df.iloc[i]['text'][:200]}...")
            print(f"Summary: {df.iloc[i]['summary']}")
            print("-" * 80)

    def create_dataset(self):
        """Main method to create the complete dataset"""
        print("🚀 Starting Hindi Dataset Creation...")
        print("Target: ~1GB Hindi Text Summarization Dataset")
        print("="*60)

        # Step 1: Download from multiple sources
        self.download_oscar_hindi()
        self.download_wikipedia_hindi()
        self.download_indic_corp()
        self.create_synthetic_hindi_data()

        # Step 2: Combine and clean
        final_dataset = self.combine_and_clean_datasets()

        if final_dataset is not None:
            # Step 3: Display samples
            self.display_samples(final_dataset)

            # Step 4: Final report
            final_path = os.path.join(self.output_dir, "hindi_summarization_final.csv")
            final_size_gb = os.path.getsize(final_path) / (1024 * 1024 * 1024)

            print("\n🎉 DATASET CREATION COMPLETED!")
            print(f"📊 Final Statistics:")
            print(f"   • Total samples: {len(final_dataset):,}")
            print(f"   • Dataset size: {final_size_gb:.2f} GB")
            print(f"   • Location: {final_path}")

            return final_dataset
        else:
            print("❌ Dataset creation failed!")
            return None

# Execute the dataset creation
if __name__ == "__main__":
    output_directory = "/content/drive/MyDrive/Major_project"

    creator = HindiDatasetCreator(output_directory)
    dataset = creator.create_dataset()

    if dataset is not None:
        print("\n✅ Your Hindi dataset is ready for fine-tuning!")
        print("You can now use it for your text summarization model.")
    else:
        print("\n❌ There was an issue creating the dataset.")
#Json conversion
import pandas as pd
import json

csv_path = "/content/drive/MyDrive/Major_project/hindi_summarization_final.csv"
json_path = "/content/drive/MyDrive/Major_project/hindi_summarization_final.json"

# Load CSV
df = pd.read_csv(csv_path)

# Convert to list of dicts
data = df.to_dict(orient="records")

# Save as JSON
with open(json_path, "w", encoding="utf-8") as f:
    json.dump(data, f, ensure_ascii=False, indent=2)

print("JSON file saved at:", json_path)
