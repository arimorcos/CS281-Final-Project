__author__ = 'arimorcos'

import scrapy
from getBookList.items import cyclopedia_entry
from getBookList.spiders.ebsco_database import ebsco_database

# TO USE SHELL
# scrapy shell "http://web.a.ebscohost.com.ezp-prod1.hul.harvard.edu/ehost/resultsadvanced?sid=51f9e4c3-5356-48d4-b835-23808edbc49e%40sessionmgr4001&vid=18&hid=4101&bquery=JN+%22Cyclopedia+of+Literary+Characters%2c+Revised+Third+Edition%22&bdata=JmRiPW1qaCZ0eXBlPTEmc2l0ZT1laG9zdC1saXZlJnNjb3BlPXNpdGU%3d"
# request.cookies = {...}
# fetch(request)


class cyclopedia_literary(ebsco_database):
    name = 'cyclopedia_literary'
    start_urls = [
        'http://web.a.ebscohost.com.ezp-prod1.hul.harvard.edu/ehost/search/advanced?sid=249051cb-0cf4-4f10-a7fd-5ee2e81c921f%40sessionmgr4001&vid=2&hid=4101'
    ]
    max_pages = 1e10
    skip_pages = range(1,11)
    username = "80832397"
    journal_name = 'JN "Cyclopedia of Literary Characters, Revised Third Edition"'

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

        self.driver.get(self.start_urls[0])

        # login and go to cyclopedia
        self.login()
        self.go_to_cylcopedia()

        next_page = True
        num_pages = 1
        while next_page:

            if num_pages in self.skip_pages:
                try:
                    next_page = self.driver.find_element_by_css_selector(
                        '#ctl00_ctl00_MainContentArea_MainContentArea_bottomMultiPage_lnkNext')
                    next_page.click()
                    num_pages += 1
                except:
                    next_page = False

                if num_pages > self.max_pages + 1:
                    break

                continue

            # get page body
            get_body = True
            while get_body:
                body = self.driver.page_source
                link_list = scrapy.Selector(text=body).xpath(
                    '//div[@class="record-formats-wrapper externalLinks"]/span/a/@href'
                ).extract()
                if len(link_list) == 50:
                    get_body = False

            # print "Page {}: {}".format(num_pages, len(link_list))

            for link in link_list:
                title, author, characters = self.get_full_text_info(link)
                # title = 'a'
                # author = 'a'
                # characters = 'a'

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

            if num_pages > self.max_pages:
                break

        self.driver.close()






