__author__ = 'arimorcos'

import scrapy
from selenium import webdriver
from getBookList.items import cyclopedia_entry
import getpass, re, time

# TO USE SHELL
# scrapy shell "http://web.a.ebscohost.com.ezp-prod1.hul.harvard.edu/ehost/resultsadvanced?sid=51f9e4c3-5356-48d4-b835-23808edbc49e%40sessionmgr4001&vid=18&hid=4101&bquery=JN+%22Cyclopedia+of+Literary+Characters%2c+Revised+Third+Edition%22&bdata=JmRiPW1qaCZ0eXBlPTEmc2l0ZT1laG9zdC1saXZlJnNjb3BlPXNpdGU%3d"
# request.cookies = {...}
# fetch(request)

class cyclopedia_literary(scrapy.Spider):
    name = 'cyclopedia_literary'
    start_urls = [
        'http://web.a.ebscohost.com.ezp-prod1.hul.harvard.edu/ehost/search/advanced?sid=51f9e4c3-5356-48d4-b835-23808edbc49e%40sessionmgr4001&vid=306&hid=4101'
    ]

    def __init__(self):
        self.pw = getpass.getpass('Enter password: ')
        self.driver = webdriver.Firefox()

    def login(self):

        # username
        element = self.driver.find_element_by_xpath('//*[@id="username"]')
        element.send_keys("80832397")

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
        element.send_keys('JN "Cyclopedia of Literary Characters, Revised Third Edition"')

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

    def parse(self, response):
        # cookies={"hulaccess": "1.3|209.6.60.59|20151101223537EST|pin|80832397|harvard|FAS-102981.HMS-103040.HMS-100151.FAS.FGS|GRAD.OFFI|2934|hul-prod",
        #          "hulaccess2_prod": "eWbxIkDP1qyKN0iQ7GikUxxfNmdEqF3E7ovdq9zDfjD8w77vOFDNE/5AqG/CedYhSRt8wmv8OqB+YbFQ67NVfyBoo0PssLP5otwdTAWuYHg=",
        #          "user_OpenURL": "http://sfx.hul.harvard.edu:80/sfx_local/",
        #          "ezproxyezpprod1": "qPRIAvBDBbyjmOK",
        #          "BIGipServersdc-web_80": "505545738.20480.0000",
        #          "_ga": "GA1.8.1317565079.1446420959",
        #          "__atuvc": "1%7C44",
        #          "__atuvs": "5636a1dfd90b070e000"}

        self.driver.get(self.start_urls[0])

        # login and go to cyclopedia
        self.login()
        self.go_to_cylcopedia()


        next_page = True
        max_pages = 1e10
        num_pages = 1
        while next_page:

            # get page body
            body = self.driver.page_source

            # get all links
            link_list = scrapy.Selector(text=body).xpath(
                '//div[@class="record-formats-wrapper externalLinks"]/span/a/@href'
            ).extract()

            for link in link_list:
                title, author, characters = self.get_full_text_info(link)

                # time.sleep(0.4)

                item = cyclopedia_entry()
                item['title'] = title
                item['author'] = author
                item['characters'] = characters

                yield item

            try:
                next_page = self.driver.find_element_by_css_selector(
                    '#ctl00_ctl00_MainContentArea_MainContentArea_bottomMultiPage_lnkNext')
                next_page.click()
                num_pages += 1
            except:
                next_page = False
                continue

            if num_pages > max_pages:
                break

        self.driver.close()






