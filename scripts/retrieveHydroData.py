import requests

# Author: Marco Maneta

class retrieve_streamflows(object):

    def __init__(self,
                 format='json',
                 site=None,
                 stateCD=None,
                 huc=None,
                 bBox=None,
                 countCD=None,
                 period=None,
                 startDT=None,
                 endDT=None,
                 parameterCD='00060',
                 siteStatus=None):
        """
        
        :param format: 
        :param site: 
        :param stateCD: 
        :param huc: 
        :param bBox: 
        :param countCD: 
        :param period: 
        :param startDT: 
        :param endDT: 
        :param parameterCD: variable to be retrieved. Default 00060 (discharge cfs)
        :param siteStatus: ['all'|'active'|'inactive'] default all 
        """

        self.baseurl = 'http://waterservices.usgs.gov/nwis/dv/'
        self.format = format
        self.site = site
        self.startDT = startDT
        self.endDT = endDT
        self.parameterCD = parameterCD

    def retrieve(self):
        return requests.get(self.baseurl, self.params())

    def params(self):
        return \
            {
                'format': self.format,
                'site': self.site,
                'startDT': self.startDT,
                'endDT': self.endDT,
                'parameterCD': self.parameterCD
            }

    def retrieveQ(self, site, startDT=None, endDT=None):
        self.site = site
        self.startDT = startDT
        self.endDT = endDT
        print self.params()
        return self.retrieve()



