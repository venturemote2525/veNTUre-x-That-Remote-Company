import { Text, ThemedSafeAreaView, View } from '@/components/Themed';
import { useEffect, useRef, useState } from 'react';
import { ScrollView, Dimensions, Pressable, Animated } from 'react-native';
import { LineChart } from 'react-native-chart-kit';
import { AnimatedPressable } from '@/components/AnimatedComponents';
import { Colors } from '@/constants/Colors';
import TabToggle from '@/components/TabToggle';
import { useRouter } from 'expo-router';
import MaterialIcons from '@expo/vector-icons/MaterialIcons';
import FontAwesome6 from '@expo/vector-icons/FontAwesome6';
import ScrollChart from '@/components/Body/ScrollChart';

const TABS = ['weight', 'BMI', 'body_fat'];
type RANGE = 'weekly' | 'monthly';

const screenWidth = Dimensions.get('window').width;

export default function BodyScreen() {
  const [screen, setScreen] = useState<(typeof TABS)[number]>('weight');
  const [range, setRange] = useState<RANGE>('weekly');
  const [selectedMetric, setSelectedMetric] = useState<
    'weight' | 'bmi' | 'bodyFat'
  >('weight');
  const router = useRouter();

  const metrics = [
    {
      id: 1,
      type: 'weight',
      value: 68,
      unit: 'kg',
      trend: '↗️',
      change: '+0.5',
    },
    { id: 2, type: 'bmi', value: 22.1, unit: '', trend: '→', change: '0.0' },
    {
      id: 3,
      type: 'bodyFat',
      value: 18.5,
      unit: '%',
      trend: '↘️',
      change: '-0.3',
    }, // camelCase fixed
  ];

  const graphData = {
    weight: [67.5, 67.8, 68.0, 68.2, 68.1, 68.0, 68.0],
    bmi: [21.9, 22.0, 22.0, 22.1, 22.2, 22.1, 22.0],
    bodyFat: [18.0, 18.2, 18.1, 18.3, 18.5, 18.4, 18.0],
  };

  // TODO: Retrieve data from database
  const tempData = [
    { value: 68, label: '24-09-2025' },
    { value: 67.8, label: '21-09-2025' },
    { value: 68.2, label: '20-09-2025' },
  ];

  function renderScreen() {
    switch (screen) {
      case 'weight':
        return (
          <View>
            <Text>Weight</Text>
          </View>
        );
      case 'BMI':
        return (
          <View>
            <Text>BMI</Text>
          </View>
        );
      case 'body_fat':
        return (
          <View>
            <Text>Body Fat</Text>
          </View>
        );
    }
  }

  return (
    <ThemedSafeAreaView edges={['top']} className="flex-1 bg-background-0">
      {/* Tab Selector */}
      <TabToggle tabs={TABS} selectedTab={screen} onTabChange={setScreen} />

      <View className="flex-1 gap-4 p-4">
        <OverviewCard screen={screen} range={range} current={2} previous={4} />
        {/* Duration Selector*/}
        <View className="mb-6 flex-row justify-center gap-4">
          <Pressable
            onPress={() => setRange('weekly')}
            className={`flex-1 items-center rounded-full px-8 py-3 ${range === 'weekly' ? 'bg-secondary-500' : 'bg-background-0'}`}>
            <Text
              className={`font-bodySemiBold ${range === 'weekly' ? 'text-background-0' : 'text-primary-100'}`}>
              Weekly
            </Text>
          </Pressable>
          <Pressable
            onPress={() => setRange('monthly')}
            className={`flex-1 items-center rounded-full px-8 py-3 ${range === 'monthly' ? 'bg-secondary-500' : 'bg-background-0'}`}>
            <Text
              className={`font-bodySemiBold ${range === 'monthly' ? 'text-background-0' : 'text-primary-100'}`}>
              Monthly
            </Text>
          </Pressable>
        </View>
        <ScrollChart graphData={tempData} />
        {renderScreen()}
      </View>
      {/*<ScrollView*/}
      {/*  contentContainerStyle={{ paddingHorizontal: 20, paddingBottom: 30 }}*/}
      {/*  showsVerticalScrollIndicator={false}>*/}
      {/*  /!* Metrics Cards *!/*/}
      {/*  <ScrollView*/}
      {/*    horizontal*/}
      {/*    showsHorizontalScrollIndicator={false}*/}
      {/*    className="mb-6 mt-6 flex-row gap-4">*/}
      {/*    {metrics.map(item => (*/}
      {/*      <AnimatedPressable*/}
      {/*        key={item.id}*/}
      {/*        onPress={() => setSelectedMetric(item.type as any)}*/}
      {/*        scaleAmount={0.95}>*/}
      {/*        <View*/}
      {/*          className={`rounded-xl p-4 ${*/}
      {/*            selectedMetric === item.type*/}
      {/*              ? 'bg-secondary-500'*/}
      {/*              : 'border border-primary-100 bg-background-0'*/}
      {/*          }`}*/}
      {/*          style={{*/}
      {/*            width: 140,*/}
      {/*            height: 120,*/}
      {/*            shadowColor: '#000',*/}
      {/*            shadowOffset: { width: 0, height: 2 },*/}
      {/*            shadowOpacity: 0.1,*/}
      {/*            shadowRadius: 3,*/}
      {/*            elevation: 3,*/}
      {/*          }}>*/}
      {/*          <View className="h-full items-center justify-center">*/}
      {/*            <Text*/}
      {/*              className={`font-bodyBold text-body2 ${selectedMetric === item.type ? 'text-background-0' : 'text-typography-800'}`}>*/}
      {/*              {item.type.toUpperCase()}*/}
      {/*            </Text>*/}
      {/*            <View className="mt-2 flex-row items-end gap-1">*/}
      {/*              <Text*/}
      {/*                style={{*/}
      {/*                  fontSize: 24,*/}
      {/*                  fontWeight: 'bold',*/}
      {/*                  color:*/}
      {/*                    selectedMetric === item.type*/}
      {/*                      ? 'white'*/}
      {/*                      : Colors.light.colors.primary[500],*/}
      {/*                }}>*/}
      {/*                {item.value.toFixed(1)}*/}
      {/*              </Text>*/}
      {/*              <Text*/}
      {/*                className={`text-body3 font-body ${selectedMetric === item.type ? 'text-background-0/90' : 'text-typography-600'} mb-1`}>*/}
      {/*                {item.unit}*/}
      {/*              </Text>*/}
      {/*            </View>*/}
      {/*            <View className="mt-2 flex-row items-center gap-1">*/}
      {/*              <Text*/}
      {/*                className={*/}
      {/*                  selectedMetric === item.type*/}
      {/*                    ? 'text-background-0'*/}
      {/*                    : 'text-typography-800'*/}
      {/*                }>*/}
      {/*                {item.trend}*/}
      {/*              </Text>*/}
      {/*              <Text*/}
      {/*                className={`text-body3 font-body ${selectedMetric === item.type ? 'text-background-0/90' : 'text-typography-600'}`}>*/}
      {/*                {item.change}*/}
      {/*              </Text>*/}
      {/*            </View>*/}
      {/*          </View>*/}
      {/*        </View>*/}
      {/*      </AnimatedPressable>*/}
      {/*    ))}*/}
      {/*  </ScrollView>*/}

      {/*  /!* View Toggle *!/*/}

      {/*  /!* Chart *!/*/}
      {/*  <View*/}
      {/*    className="mb-5 rounded-xl bg-background-0 p-5"*/}
      {/*    style={{*/}
      {/*      shadowColor: '#000',*/}
      {/*      shadowOffset: { width: 0, height: 2 },*/}
      {/*      shadowOpacity: 0.1,*/}
      {/*      shadowRadius: 3,*/}
      {/*      elevation: 3,*/}
      {/*    }}>*/}
      {/*    <Text className="mb-4 text-center font-bodyBold text-body1 text-secondary-500">*/}
      {/*      {selectedMetric === 'weight'*/}
      {/*        ? 'Weight Trend'*/}
      {/*        : selectedMetric === 'bmi'*/}
      {/*          ? 'BMI Progress'*/}
      {/*          : 'Body Fat %'}*/}
      {/*    </Text>*/}
      {/*    <LineChart*/}
      {/*      data={{*/}
      {/*        labels:*/}
      {/*          range === 'weekly'*/}
      {/*            ? ['Mon', 'Tue', 'Wed', 'Thu', 'Fri', 'Sat', 'Sun']*/}
      {/*            : ['Week 1', 'Week 2', 'Week 3', 'Week 4'],*/}
      {/*        datasets: [*/}
      {/*          {*/}
      {/*            data: (range === 'weekly'*/}
      {/*              ? graphData[selectedMetric].slice(0, 7)*/}
      {/*              : graphData[selectedMetric].slice(0, 4)*/}
      {/*            ).map(v => parseFloat(v.toFixed(1))),*/}
      {/*          },*/}
      {/*        ],*/}
      {/*      }}*/}
      {/*      width={screenWidth - 80}*/}
      {/*      height={220}*/}
      {/*      yAxisSuffix={*/}
      {/*        selectedMetric === 'weight'*/}
      {/*          ? 'kg'*/}
      {/*          : selectedMetric === 'bodyFat'*/}
      {/*            ? '%'*/}
      {/*            : ''*/}
      {/*      }*/}
      {/*      chartConfig={{*/}
      {/*        backgroundColor: '#ffffff',*/}
      {/*        backgroundGradientFrom: '#ffffff',*/}
      {/*        backgroundGradientTo: '#ffffff',*/}
      {/*        decimalPlaces: 1,*/}
      {/*        color: () => Colors.light.colors.primary[500],*/}
      {/*        labelColor: () => Colors.light.colors.primary[700],*/}
      {/*        style: { borderRadius: 16 },*/}
      {/*        propsForDots: {*/}
      {/*          r: '6',*/}
      {/*          strokeWidth: '2',*/}
      {/*          stroke: Colors.light.colors.primary[500],*/}
      {/*        },*/}
      {/*        propsForBackgroundLines: {*/}
      {/*          stroke: Colors.light.colors.secondary[200],*/}
      {/*          strokeWidth: 1,*/}
      {/*        },*/}
      {/*      }}*/}
      {/*      bezier*/}
      {/*      style={{ marginVertical: 8, borderRadius: 16 }}*/}
      {/*    />*/}
      {/*  </View>*/}

      {/*  /!* Progress Message *!/*/}
      {/*  <View className="rounded-xl border border-secondary-200 bg-secondary-50 p-4">*/}
      {/*    <Text className="text-center font-body text-body2 text-typography-800">*/}
      {/*      You've maintained a healthy {selectedMetric} range this week. Keep*/}
      {/*      it up!*/}
      {/*    </Text>*/}
      {/*  </View>*/}
      {/*</ScrollView>*/}
      <Pressable
        className="button-rounded m-4"
        onPress={() => router.push('/(body)/all-logs')}>
        <Text className="button-rounded-text">View All Logs</Text>
      </Pressable>
    </ThemedSafeAreaView>
  );
}

export function OverviewCard({
  screen,
  range,
  current,
  previous,
}: {
  screen: (typeof TABS)[number];
  range: RANGE;
  current: number;
  previous: number;
}) {
  // Trend Calculation
  const difference = current - previous;
  const trendIcon =
    difference > 0 ? (
      <FontAwesome6
        name="arrow-trend-up"
        size={28}
        color={Colors.light.colors.secondary[500]}
      />
    ) : difference < 0 ? (
      <FontAwesome6
        name="arrow-trend-down"
        size={28}
        color={Colors.light.colors.secondary[500]}
      />
    ) : (
      <MaterialIcons
        name="trending-flat"
        size={28}
        color={Colors.light.colors.secondary[500]}
      />
    );
  const trendText = difference !== 0 ? Math.abs(difference).toFixed(1) : '-';

  // Animate flex values
  const flexAnim = useRef(new Animated.Value(1)).current; // start at flex-1
  useEffect(() => {
    Animated.timing(flexAnim, {
      toValue: range === 'weekly' ? 1 : 2, // Animate flex value
      duration: 300,
      useNativeDriver: false,
    }).start();
  }, [range, flexAnim]);

  return (
    <View className="w-full flex-row gap-4">
      {/* Current */}
      <Animated.View
        style={{ flex: flexAnim }}
        className="items-center justify-center gap-2 rounded-2xl bg-background-0 p-4">
        <Text className="text-primary-300">Current</Text>
        <Text className="font-bodySemiBold text-body1 text-secondary-500">
          50kg
        </Text>
      </Animated.View>

      {/* Trend */}
      <Animated.View
        style={{ flex: Animated.multiply(flexAnim, 2) }}
        className="items-center justify-center gap-2 rounded-2xl bg-background-0 p-4">
        <Text className="text-primary-300">
          Trend from last {range === 'weekly' ? 'week' : 'month'}
        </Text>
        <View className="flex-row items-center gap-2">
          {trendIcon}
          <Text className="font-bodySemiBold text-body1 text-secondary-500">
            {trendText}
          </Text>
        </View>
      </Animated.View>
    </View>
  );
}
