#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jan 27 11:06:53 2021

@author: nuvilabs
"""

class NuviS3KeySensor(S3KeySensor):

        
    def poke(self, context):
        to_return = super().poke(context)

        if to_return:
            hook = S3Hook(aws_conn_id=self.aws_conn_id)
            keys = hook.list_keys(self.bucket_name)

            check_key = lambda key: 'ID' in key                              
            check_key_v = np.vectorize(check_key)
            keys = np.asarray(keys)
            idx = check_key_v(keys)
            ID_keys = keys[idx].tolist()
            to_return = len(ID_keys) != 0
            if to_return:self.xcom_push(context, key="IDS", value = ID_keys)

        return to_return