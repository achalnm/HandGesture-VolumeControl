import argparse
import os

from icrawler.builtin import BingImageCrawler

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


def download(folder, queries, per_query):
    os.makedirs(folder, exist_ok=True)
    for query in queries:
        print(f"Fetching: {query}")
        crawler = BingImageCrawler(
            storage={'root_dir': folder},
            feeder_threads=1,
            parser_threads=1,
            downloader_threads=4,
        )
        crawler.crawl(keyword=query, max_num=per_query, min_size=(80, 80))


def count_images(folder):
    exts = {'.jpg', '.jpeg', '.png', '.webp', '.bmp'}
    return sum(1 for f in os.listdir(folder) if os.path.splitext(f.lower())[1] in exts)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--folder1',   default='data/images/thumbs_up')
    parser.add_argument('--folder2',   default='data/images/thumbs_down')
    parser.add_argument('--per_query', type=int, default=60)
    args = parser.parse_args()

    download(args.folder1, QUERIES['thumbs_up'],   args.per_query)
    download(args.folder2, QUERIES['thumbs_down'], args.per_query)

    print(f"\nfolder1: {count_images(args.folder1)} images")
    print(f"folder2: {count_images(args.folder2)} images")


if __name__ == '__main__':
    main()
