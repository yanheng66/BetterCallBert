import os
import gc
from loader import extract_text_from_images
from index_builder_faiss import build_faiss_index

def build_all_uscode_faiss_indexes(pdf_dir="data/uscode", output_dir="vector_store_faiss"):
    os.makedirs(output_dir, exist_ok=True)

    for fname in sorted(os.listdir(pdf_dir)):
        if fname.endswith(".pdf"):
            title_id = fname.replace(".pdf", "").lower()
            pdf_path = os.path.join(pdf_dir, fname)
            persist_path = os.path.join(output_dir, title_id)

            if os.path.exists(persist_path):
                print(f"✅ 已存在向量库，跳过：{title_id}")
                continue

            try:
                print(f"\n📘 Building FAISS index for {title_id} ...")
                doc = extract_text_from_images(pdf_path)
                doc.metadata = {"source": title_id}
                build_faiss_index([doc], persist_dir=persist_path)
                gc.collect()
            except Exception as e:
                print(f"❌ 构建失败 {title_id}：{e}")

if __name__ == "__main__":
    build_all_uscode_faiss_indexes()
