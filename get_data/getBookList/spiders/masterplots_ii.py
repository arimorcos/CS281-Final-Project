__author__ = 'arimorcos'

import scrapy
from lxml import html
from getBookList.items import book_summary
from getBookList.spiders.ebsco_database import ebsco_database

# TO USE SHELL
# scrapy shell "http://web.a.ebscohost.com.ezp-prod1.hul.harvard.edu/ehost/resultsadvanced?sid=51f9e4c3-5356-48d4-b835-23808edbc49e%40sessionmgr4001&vid=18&hid=4101&bquery=JN+%22Cyclopedia+of+Literary+Characters%2c+Revised+Third+Edition%22&bdata=JmRiPW1qaCZ0eXBlPTEmc2l0ZT1laG9zdC1saXZlJnNjb3BlPXNpdGU%3d"
# request.cookies = {...}
# fetch(request)

class masterplots_ii(ebsco_database):
    name = 'masterplots_ii'
    start_urls = [
        'http://web.a.ebscohost.com.ezp-prod1.hul.harvard.edu/ehost/search/advanced?sid=249051cb-0cf4-4f10-a7fd-5ee2e81c921f%40sessionmgr4001&vid=2&hid=4101'
    ]
    max_pages = 1e10
    skip_pages = range(1,130)
    username = "80832397"
    journal_name = '"Masterplots II"'

    def remove_table_of_contents(self, body):

        doc = html.fromstring(body)
        for el in doc.xpath('//div[@class="html-ft-toc" and @data-auto="html_toc"]'):
            el.drop_tree()
        result = html.tostring(doc)
        return result

    def get_summary_selectors(self,body):
        # limit to possible descriptions
        summary_selectors = scrapy.Selector(text=body).xpath(
            """
            //p[preceding::span/a[contains(@title,"The Story")]]
            [count(.|//p[following-sibling::span/a[contains(@title,"Critical Evaluation")]])
            =
            count(//p[following-sibling::span/a[contains(@title,"Critical Evaluation")]])]
            """)
        if not summary_selectors:
            summary_selectors = scrapy.Selector(text=body).xpath(
                """
                //p[preceding::span/a[contains(@title,"The Work")]]
                [count(.|//p[following-sibling::span/a[contains(@title,"Critical Evaluation")]])
                =
                count(//p[following-sibling::span/a[contains(@title,"Critical Evaluation")]])]
                """)
        if not summary_selectors:
            summary_selectors = scrapy.Selector(text=body).xpath(
                """
                //p[preceding::span/a[contains(@title,"The Work")]]
                [count(.|//p[following-sibling::span/a[contains(@title,"Further Reading")]])
                =
                count(//p[following-sibling::span/a[contains(@title,"Further Reading")]])]
                """)
        if not summary_selectors:
            summary_selectors = scrapy.Selector(text=body).xpath(
                """
                //p[preceding::span/a[contains(@title,"The Story")]]
                [count(.|//p[following-sibling::span/a[contains(@title,"Further Reading")]])
                =
                count(//p[following-sibling::span/a[contains(@title,"Further Reading")]])]
                """)
        if not summary_selectors:
            summary_selectors = scrapy.Selector(text=body).xpath(
                """
                //p[preceding::span/a[contains(@title,"The Novel")]]
                [count(.|//p[following-sibling::span/a[contains(@title,"The Characters")]])
                =
                count(//p[following-sibling::span/a[contains(@title,"The Characters")]])]
                """)
        if not summary_selectors:
            summary_selectors = scrapy.Selector(text=body).xpath(
                """
                //p[preceding::span/a[contains(@title,"Form and Content")]]
                [count(.|//p[following-sibling::span/a[contains(@title,"Analysis")]])
                =
                count(//p[following-sibling::span/a[contains(@title,"Analysis")]])]
                """)
        if not summary_selectors:
            summary_selectors = scrapy.Selector(text=body).xpath(
                """
                //p[preceding::span/a[contains(@title,"The Play")]]
                [count(.|//p[following-sibling::span/a[contains(@title,"Themes and Meanings")]])
                =
                count(//p[following-sibling::span/a[contains(@title,"Themes and Meanings")]])]
                """)
        if not summary_selectors:
            summary_selectors = scrapy.Selector(text=body).xpath(
                """
                //p[preceding::span/a[contains(@title,"The Stories")]]
                [count(.|//p[following-sibling::span/a[contains(@title,"The Characters")]])
                =
                count(//p[following-sibling::span/a[contains(@title,"The Characters")]])]
                """)
        if not summary_selectors:
            summary_selectors = scrapy.Selector(text=body).xpath(
                """
                //p[preceding::span/a[contains(@title,"The Poems")]]
                [count(.|//p[following-sibling::span/a[contains(@title,"Themes and Meanings")]])
                =
                count(//p[following-sibling::span/a[contains(@title,"Themes and Meanings")]])]
                """)
        if not summary_selectors:
            summary_selectors = scrapy.Selector(text=body).xpath(
                """
                //p[preceding::span/a[contains(@title,"The Story")]]
                [count(.|//p[following-sibling::span/a[contains(@title,"Themes and Meanings")]])
                =
                count(//p[following-sibling::span/a[contains(@title,"Themes and Meanings")]])]
                """)
        if not summary_selectors:
            summary_selectors = scrapy.Selector(text=body).xpath(
                """
                //p[preceding::span/a[contains(@title,"Overview")]]
                [count(.|//p[following-sibling::span/a[contains(@title,"Christian Themes")]])
                =
                count(//p[following-sibling::span/a[contains(@title,"Christian Themes")]])]
                """)
        if not summary_selectors:
            summary_selectors = scrapy.Selector(text=body).xpath(
                """
                //p[preceding::span/a[contains(@title,"Overview")]]
                [count(.|//p[following-sibling::span/a[contains(@title,"The Poem")]])
                =
                count(//p[following-sibling::span/a[contains(@title,"Forms and Devices")]])]
                """)

        return summary_selectors

    def get_full_text_summary(self, link):

        # follow link
        self.driver.get(link)

        # get page body
        get_body = True
        while get_body:
            body = self.driver.page_source
            temp = scrapy.Selector(text=body).xpath('//h2[@data-auto="local_abody_title"]/text()').extract()
            if temp:
                get_body = False

        # remove table of contents
        body = self.remove_table_of_contents(body)

        # get listed author and title
        try:
            title = scrapy.Selector(text=body).xpath('//h2[@data-auto="local_abody_title"]/text()').extract()[0]
        except:
            raise
        author = scrapy.Selector(text=body).xpath('//section[@class="full-text-content textToSpeechDataContainer"]/p/strong/text()').extract()[0]

        summary_selectors = self.get_summary_selectors(body)

        # get character descriptions
        summary = "".join(summary_selectors.xpath('.//text()').extract())
        summary = summary.split("Essay by:")[0]

        # if not summary:
        #     print "Work: {}, author: {} ||| Summary not found".format(title, author)
        #     print '\a'
        #     return

        # go back
        self.driver.back()

        return title, author, summary

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
                    get_next_page = True
                    while get_next_page:
                        body = self.driver.page_source
                        next_page = self.driver.find_element_by_css_selector(
                                    '#ctl00_ctl00_MainContentArea_MainContentArea_bottomMultiPage_lnkNext')
                        if next_page:
                            get_next_page = False
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
                if len(link_list) > 0:
                    get_body = False

            # print "Page {}: {}".format(num_pages, len(link_list))

            for link in link_list:
                title, author, summary = self.get_full_text_summary(link)
                # title = 'a'
                # author = 'a'
                # characters = 'a'
                if not summary:
                    continue

                item = book_summary()
                item['title'] = title
                item['author'] = author
                item['summary'] = summary
                item['summary_type'] = "masterplots_ii"

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






