import os
import nltk
from agnews_model import predict_single

# 設置 NLTK 資料路徑
base_dir = os.getcwd()
nltk_data_dir = os.path.join(base_dir, "nltk_data")
if not os.path.exists(nltk_data_dir):
    os.makedirs(nltk_data_dir)
nltk.data.path.append(nltk_data_dir)

 
# 展示預測
def main():
    # 示例新聞文本
    sample_texts = [
        "Tech company launches new AI-powered smartphone with advanced features.",
        "Stock market crashes as global economic concerns rise.",
        "New soccer season starts with thrilling matches and surprises."
    ]

    # 執行預測
    for text in sample_texts:
        idx, label = predict_single(
            text=text,
            model_path='config/agnews_model.pth',
            vocab_path='config/vocab.pkl',
            class_map_path='config/class_map.pkl',
            max_len=100
        )
        print(f"Text: {text}")
        print(f"Predicted Index: {idx}, Label: {label}\n")

if __name__ == "__main__":
    main()

''' Output Recording

Text: Tech company launches new AI-powered smartphone with advanced features.
Predicted Index: 3, Label: Sci/Tech

Text: Stock market crashes as global economic concerns rise.
Predicted Index: 2, Label: Business

Text: New soccer season starts with thrilling matches and surprises.
Predicted Index: 0, Label: World


'''