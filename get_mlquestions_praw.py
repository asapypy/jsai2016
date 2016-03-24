#!/usr/bin/env python
# encoding: utf-8

from __future__ import print_function
import six
import sys
import praw
from pprint import pprint

useragent = "Karma breakdown 1.0 by /u/_Daimon_"
reddit = praw.Reddit(useragent)

# pprint(vars(reddit))
# pprint(dir(reddit))
# print(dir(reddit.user))
# print(dir(reddit))

def scrape(limit = 20):
    subredditList = reddit.get_subreddit('mlquestions').get_hot(limit = limit)
    for n, subreddit in enumerate(subredditList):
        print('Question= ', n, end=' ')
        print('title=', subreddit.title)
# print(subreddit.num_comments, 
#              repr(subreddit.domain), 
#              repr(subreddit.fullname), 
#              subreddit.num_reports, 
#              repr(subreddit.selftext)) 
#              # subreddit.selftext_html)
        if subreddit.num_comments > 1:
            print('author=', subreddit.author)
            print('selttext= ', subreddit.selftext.encode('utf-8'))
            print('num_comments=', subreddit.num_comments)
            print('len(subreddit.comments)=', len(subreddit.comments))
            for i, com in enumerate(subreddit.comments):
                print('comment no.=', i, '--> ', subreddit.comments[i].body.encode('utf-8'))
        else:
            print('no comments')
        print('-------------')

def main(num):
    scrape(limit = num)

if __name__ == '__main__':
    args = sys.argv 
    if len(args) > 1:
        print(args[1])
        count = int(args[1])
    else:
        count = 10
    main(count)
