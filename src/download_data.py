import argparse
import os

# pip install icrawler
from icrawler.builtin import BingImageCrawler

# Search queries per class. Multiple queries = more variety in backgrounds,
# lighting and hand shapes, which is exactly what improves generalisation.
QUERIES = {
    'thumbs_up': [
        'thumbs up hand gesture',
        'thumb up sign hand photo',
        'thumbs up hand white background',
        'person thumbs up gesture',
    ],
    'thumbs_down': [
        'thumbs down hand gesture',
        'thumb down sign hand photo',
        'thumbs down hand white background',
        'person thumbs down gesture',
    ],
}


def download_images(folder, queries, per_query):
    os.makedirs(folder, exist_ok=True)
    for query in queries:
        print(f"\nDownloading: '{query}' -> {folder}/")
        crawler = BingImageCrawler(
            storage={'root_dir': folder},
            feeder_threads=1,
            parser_threads=1,
            downloader_threads=4,
        )
        crawler.crawl(
            keyword=query,
            max_num=per_query,
            min_size=(80, 80),   # skip tiny/broken images
        )


def main():
    parser = argparse.ArgumentParser(description='Download gesture images from Bing')
    parser.add_argument('--folder1',   type=str, default='folder1',
                        help='Output folder for thumbs-up images')
    parser.add_argument('--folder2',   type=str, default='folder2',
                        help='Output folder for thumbs-down images')
    parser.add_argument('--per_query', type=int, default=60,
                        help='Images to download per search query (default 60)')
    args = parser.parse_args()

    total = args.per_query * len(QUERIES['thumbs_up'])
    print(f"Will download up to {total} images per class "
          f"({args.per_query} x {len(QUERIES['thumbs_up'])} queries).")
    print("Downloading thumbs-up images...")
    download_images(args.folder1, QUERIES['thumbs_up'], args.per_query)

    print("\nDownloading thumbs-down images...")
    download_images(args.folder2, QUERIES['thumbs_down'], args.per_query)

    # Count what we got
    def count_images(folder):
        exts = {'.jpg', '.jpeg', '.png', '.webp'}
        return sum(
            1 for f in os.listdir(folder)
            if os.path.splitext(f.lower())[1] in exts
        )

    n1 = count_images(args.folder1)
    n2 = count_images(args.folder2)
    print(f"\nDone.")
    print(f"  {args.folder1}: {n1} images")
    print(f"  {args.folder2}: {n2} images")
    print(f"\nNext steps:")
    print(f"  python src/preprocess.py --folder1 {args.folder1} "
          f"--folder2 {args.folder2} --output data/labels.csv")
    print(f"  python src/train.py --csv data/labels.csv --output outputs/model.h5")


if __name__ == '__main__':
    main()
