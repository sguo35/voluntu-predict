def load_data(): 
    orgs = ['5be92f95-77d1-11e6-9181-f23c9191ab2a',
    '5be933f7-77d1-11e6-9181-f23c9191ab2a',
    '5be93119-77d1-11e6-9181-f23c9191ab2a',
    '5be9291d-77d1-11e6-9181-f23c9191ab2a',
    '5be93291-77d1-11e6-9181-f23c9191ab2a',
    '0f168aac-b0de-4edf-a065-f5cff0602192',
    '5be92d8e-77d1-11e6-9181-f23c9191ab2a',
    '5be936ef-77d1-11e6-9181-f23c9191ab2a',
    '5be9358c-77d1-11e6-9181-f23c9191ab2a',
    '17b6c69c-1708-11e7-b71d-f23c9191ab2a',
    '1659dc59-d4eb-4eab-92a7-732e41f5763d',
    'f278fe21-8bb6-4d34-acd7-6d8158877535']

    dictionary = 'ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz1234567890,.-&!'

    from datetime import datetime
    from dateutil.parser import parse
    import numpy as np
    import pandas as pd
    data = pd.read_csv('./out.csv', sep=',').values

    output_x = []
    output_y = []

    for row in data:
        orgIndex = orgs.index(row[0])
        startTime = parse(row[1])
        endTime = parse(row[2])
        dayOfWeekIndex = startTime.weekday()
        startHourIndex = startTime.hour
        startTimeSec = startTime.timestamp()
        endTimeSec = endTime.timestamp()
        rowout = []
        desc = list(row[3])
        status = row[4]
        i = 0
        while i < 200:
            #orgArr = np.zeros(12, dtype='float')
            dowArr = np.zeros(7, dtype='float')
            startHrArr = np.zeros(24, dtype='float')
            durArr = np.zeros(1, dtype='float')
            charArr = np.zeros(67, dtype='float')
            
            #orgArr[orgIndex] = 1.0
            dowArr[dayOfWeekIndex] = 1.0
            startHrArr[startHourIndex] = 1.0
            durArr[0] = (endTimeSec - startTimeSec) / 36000.0 # 10 hours
            if len(desc) > i and dictionary.find(desc[i]) != -1:
                charArr[dictionary.find(desc[i])] = 1.0
            rowout.append(np.concatenate((durArr, charArr, startHrArr, dowArr)))
            #rowout.append(charArr)
            i += 1
        
        if status == 'approved':
            output_x.append(rowout)
            output_y.append([1., 0])
        else:
            j = 0
            # duplicating data is more stable for training than 100x weight for negative examples
            while j < 20:
                output_x.append(rowout)
                output_y.append([0, 1.])
                j += 1
    return np.array(output_x), np.array(output_y)
