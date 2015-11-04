from selenium import webdriver
import scrapy
import getpass
import re
from abc import ABCMeta, abstractmethod

# TO USE SHELL
# scrapy shell "http://web.a.ebscohost.com.ezp-prod1.hul.harvard.edu/ehost/resultsadvanced?sid=51f9e4c3-5356-48d4-b835-23808edbc49e%40sessionmgr4001&vid=18&hid=4101&bquery=JN+%22Cyclopedia+of+Literary+Characters%2c+Revised+Third+Edition%22&bdata=JmRiPW1qaCZ0eXBlPTEmc2l0ZT1laG9zdC1saXZlJnNjb3BlPXNpdGU%3d"
# request.cookies = {...}
# fetch(request)


class ebsco_database(scrapy.Spider):
    __metaclass__ = ABCMeta

    max_pages = 1e10
    skip_pages = []
    username = "80832397"
    journal_name = 'JN "Cyclopedia of Literary Characters, Revised Third Edition"'

    def __init__(self):
        self.pw = getpass.getpass('Enter password: ')
        self.driver = webdriver.Firefox()

    def login(self):

        # username
        element = self.driver.find_element_by_xpath('//*[@id="username"]')
        element.send_keys(self.username)

        # password
        element = self.driver.find_element_by_xpath('//*[@id="password"]')
        element.send_keys(self.pw)

        # login
        element = self.driver.find_element_by_xpath('//*[@id="submitLogin"]')
        element.click()

        # check if didn't work
        try:
            element = self.driver.find_element_by_xpath(
                '//*[@id="ctl00_ctl00_MainContentArea_MainContentArea_linkError"]')
            element.click()
        except:
            pass

    def go_to_cylcopedia(self):

        # search for cyclopedia
        element = self.driver.find_element_by_xpath('//*[@id="Searchbox1"]')
        element.send_keys(self.journal_name)

        # click
        element = self.driver.find_element_by_xpath('//*[@id="SearchButton"]')
        element.click()

    def get_author_list(self, book_info):

        # concatenate limit to fields with author name
        book_info = [x for x in book_info if re.match('.*Author Name:.*', x)]

        # get author names
        author_list = [re.match('.*((?<=Author Name: ).+)', info).groups(0)[0] for info in book_info]

        return author_list

    def get_character_descriptions(self, poss_char_descriptions):

        characters = []
        for char in poss_char_descriptions:

            # find bold tag to get name
            char_name = char.xpath('.//b/text()').extract()
            if not char_name:
                continue
            char_name = char_name[0]

            # get description
            all_text = char.xpath('.//text()').extract()
            desc_ind = [ind for ind, x
                        in enumerate(all_text)
                        if char_name == x][0] + 1
            char_description = all_text[desc_ind]


            characters.append(
                {'name': char_name,
                 'description': char_description})

        return characters

    def get_full_text_info(self, link):

        # follow link
        self.driver.get(link)

        # get page body
        get_body = True
        while get_body:
            body = self.driver.page_source
            temp = scrapy.Selector(text=body).xpath('//h2[@data-auto="local_abody_title"]/text()').extract()
            if temp:
                get_body = False


        # get listed author and title
        try:
            title = scrapy.Selector(text=body).xpath('//h2[@data-auto="local_abody_title"]/text()').extract()[0]
        except:
            raise
        author = scrapy.Selector(text=body).xpath('//section[@class="full-text-content textToSpeechDataContainer"]/p/strong/text()').extract()[0]

        # limit to possible descriptions
        # poss_char_descriptions = scrapy.Selector(text=body).xpath(
        #     """
        #     //p[preceding::span/a[@title="Characters Discussed"]]
        #     [count(.|//p[following-sibling::span/a[contains(@title,"Bibliography")]])
        #     =
        #     count(//p[following-sibling::span/a[contains(@title,"Bibliography")]])]
        #     """)
        poss_char_descriptions = scrapy.Selector(text=body).xpath(
            """
            //p[preceding::span/a[@title="Characters Discussed"]]
            """)

        # get character descriptions
        characters = self.get_character_descriptions(poss_char_descriptions)

        # go back
        self.driver.back()

        return title, author, characters

    @abstractmethod
    def parse(self, response):
        pass





