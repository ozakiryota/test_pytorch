from icrawler.builtin import BingImageCrawler

rootpath = "./data/crawing"
keyword = "lion"
max_num = 10

crawler = BingImageCrawler(storage = {"root_dir" : rootpath})
crawler.crawl(keyword=keyword, max_num=max_num)
