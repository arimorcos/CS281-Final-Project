# -*- coding: utf-8 -*-

# Define here the models for your scraped items
#
# See documentation in:
# http://doc.scrapy.org/en/latest/topics/items.html

import scrapy


class book(scrapy.Item):
    # define the fields for your item here like:
    title = scrapy.Field()
    author = scrapy.Field()


class cyclopedia_entry(scrapy.Item):
    title = scrapy.Field()
    characters = scrapy.Field()
    author = scrapy.Field()

class book_summary(scrapy.Item):
    title = scrapy.Field()
    summary = scrapy.Field()
    author = scrapy.Field()
    summary_type = scrapy.Field()