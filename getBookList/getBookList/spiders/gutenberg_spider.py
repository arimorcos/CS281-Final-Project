__author__ = 'arimorcos'

import scrapy
import re, string

from getBookList.items import book

class gutenberg_spider(scrapy.Spider):

    name = "gutenberg"
    allowed_domains = ["gutenberg.org"]
    start_urls = ["".join(["http://www.gutenberg.org/browse/titles/",letter]) for letter in string.ascii_lowercase]
    # start_urls = ["".join(["http://www.gutenberg.org/browse/titles/",letter]) for letter in "a"]

    def parse(self, response):
        language_list = response.xpath('/html/body/div[2]/div[3]/div[2]/h2/text()').extract()
        title_list = response.xpath('/html/body/div[2]/div[3]/div[2]/h2/a/text()[1]').extract()
        author_list = response.xpath('/html/body/div[2]/div[3]/div[2]/p/a/text()').extract()

        for (lang, title, author) in zip(language_list, title_list, author_list):

            # check if english
            if re.match(".*English.*", lang) is not None:
                if author == "Unknown" or author == "Various":
                    continue

                item = book()
                item['title'] = title
                item['author'] = author

                yield item


