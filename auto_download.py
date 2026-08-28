#!/usr/bin/env python3
import os
import sys

try:
    from icrawler.builtin import GoogleImageCrawler
except ImportError:
    print("❌ icrawler 라이브러리가 설치되어 있지 않습니다.")
    print("👉 터미널에서 다음 명령어를 실행하여 설치해주세요: pip install icrawler")
    sys.exit(1)

# 📁 수작업 코드와 동일한 데이터셋 폴더 경로
base_dir = os.path.dirname(os.path.abspath(__file__))
save_folder = os.path.join(base_dir, 'jb_light_dataset')

if not os.path.exists(save_folder):
    os.makedirs(save_folder)

# 크롤러 설정
google_crawler = GoogleImageCrawler(
    storage={'root_dir': save_folder}
)

# 검색할 키워드와 다운받을 사진 개수 설정
search_keyword = '제이비솔루션 신호차'
max_images = 100

print(f"🚀 '{search_keyword}' 키워드로 최대 {max_images}장 이미지 크롤링 시작...")
print(f"📂 저장 폴더: {save_folder}")

google_crawler.crawl(keyword=search_keyword, max_num=max_images)

print("=" * 60)
print("✅ 다운로드가 완료되었습니다!")
print("⚠️ [중요] 'jb_light_dataset' 폴더를 열어 신호등과 관련 없는 사진들을 반드시 직접 삭제(정제)해주세요.")
print("=" * 60)
