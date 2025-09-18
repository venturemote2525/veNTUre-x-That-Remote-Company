//
//  ICKitchenDeviceInfo.h
//  ICDeviceManager
//
//  Created by Symons on 2020/4/29.
//  Copyright © 2020 Symons. All rights reserved.
//

#import <Foundation/Foundation.h>
#import "ICDeviceInfo.h"

NS_ASSUME_NONNULL_BEGIN

@interface ICKitchenDeviceInfo : ICDeviceInfo

/**
 秤支持的功能
 */
@property(nonatomic, strong) NSArray<NSNumber *> *supportFuns;

/**
 支持秤上显示的营养数据类型
 */
@property(nonatomic, strong) NSArray<NSNumber *> *supportDataTypes;

/**
 当前秤上的历史数据数量
 */
@property(nonatomic, assign) NSUInteger historyCount;

@end

NS_ASSUME_NONNULL_END
