import { Text, ThemedSafeAreaView, View } from '@/components/Themed';
import { useState } from 'react';
import { ScrollView, Dimensions, Pressable } from 'react-native';
import { LineChart } from 'react-native-chart-kit';
import { AnimatedPressable } from '@/components/AnimatedComponents';
import { Colors } from '@/constants/Colors';
import TabToggle from '@/components/TabToggle';
import { useRouter } from 'expo-router';

const TABS = ['weight', 'BMI', 'Body Fat'];

const screenWidth = Dimensions.get('window').width;

export default function BodyScreen() {
  const [screen, setScreen] = useState<(typeof TABS)[number]>('weight');
  const [graphView, setGraphView] = useState<'weekly' | 'monthly'>('weekly');
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

  return (
    <ThemedSafeAreaView edges={['top']} className="flex-1 bg-background-0">
      <TabToggle tabs={TABS} selectedTab={screen} onTabChange={setScreen} />
      <ScrollView
        contentContainerStyle={{ paddingHorizontal: 20, paddingBottom: 30 }}
        showsVerticalScrollIndicator={false}>
        {/* Metrics Cards */}
        <ScrollView
          horizontal
          showsHorizontalScrollIndicator={false}
          className="mb-6 mt-6 flex-row gap-4">
          {metrics.map(item => (
            <AnimatedPressable
              key={item.id}
              onPress={() => setSelectedMetric(item.type as any)}
              scaleAmount={0.95}>
              <View
                className={`rounded-xl p-4 ${
                  selectedMetric === item.type
                    ? 'bg-secondary-500'
                    : 'border border-primary-100 bg-background-0'
                }`}
                style={{
                  width: 140,
                  height: 120,
                  shadowColor: '#000',
                  shadowOffset: { width: 0, height: 2 },
                  shadowOpacity: 0.1,
                  shadowRadius: 3,
                  elevation: 3,
                }}>
                <View className="h-full items-center justify-center">
                  <Text
                    className={`font-bodyBold text-body2 ${selectedMetric === item.type ? 'text-background-0' : 'text-typography-800'}`}>
                    {item.type.toUpperCase()}
                  </Text>
                  <View className="mt-2 flex-row items-end gap-1">
                    <Text
                      style={{
                        fontSize: 24,
                        fontWeight: 'bold',
                        color:
                          selectedMetric === item.type
                            ? 'white'
                            : Colors.light.colors.primary[500],
                      }}>
                      {item.value.toFixed(1)}
                    </Text>
                    <Text
                      className={`text-body3 font-body ${selectedMetric === item.type ? 'text-background-0/90' : 'text-typography-600'} mb-1`}>
                      {item.unit}
                    </Text>
                  </View>
                  <View className="mt-2 flex-row items-center gap-1">
                    <Text
                      className={
                        selectedMetric === item.type
                          ? 'text-background-0'
                          : 'text-typography-800'
                      }>
                      {item.trend}
                    </Text>
                    <Text
                      className={`text-body3 font-body ${selectedMetric === item.type ? 'text-background-0/90' : 'text-typography-600'}`}>
                      {item.change}
                    </Text>
                  </View>
                </View>
              </View>
            </AnimatedPressable>
          ))}
        </ScrollView>

        {/* View Toggle */}
        <View className="mb-6 flex-row justify-center gap-4">
          <AnimatedPressable
            onPress={() => setGraphView('weekly')}
            className={`rounded-full px-6 py-3 ${graphView === 'weekly' ? 'bg-secondary-500' : 'border border-primary-200 bg-background-0'}`}
            scaleAmount={0.95}>
            <Text
              className={`font-bodySemiBold ${graphView === 'weekly' ? 'text-background-0' : 'text-secondary-500'}`}>
              Weekly
            </Text>
          </AnimatedPressable>
          <AnimatedPressable
            onPress={() => setGraphView('monthly')}
            className={`rounded-full px-6 py-3 ${graphView === 'monthly' ? 'bg-secondary-500' : 'border border-primary-200 bg-background-0'}`}
            scaleAmount={0.95}>
            <Text
              className={`font-bodySemiBold ${graphView === 'monthly' ? 'text-background-0' : 'text-secondary-500'}`}>
              Monthly
            </Text>
          </AnimatedPressable>
        </View>

        {/* Chart */}
        <View
          className="mb-5 rounded-xl bg-background-0 p-5"
          style={{
            shadowColor: '#000',
            shadowOffset: { width: 0, height: 2 },
            shadowOpacity: 0.1,
            shadowRadius: 3,
            elevation: 3,
          }}>
          <Text className="mb-4 text-center font-bodyBold text-body1 text-secondary-500">
            {selectedMetric === 'weight'
              ? 'Weight Trend'
              : selectedMetric === 'bmi'
                ? 'BMI Progress'
                : 'Body Fat %'}
          </Text>
          <LineChart
            data={{
              labels:
                graphView === 'weekly'
                  ? ['Mon', 'Tue', 'Wed', 'Thu', 'Fri', 'Sat', 'Sun']
                  : ['Week 1', 'Week 2', 'Week 3', 'Week 4'],
              datasets: [
                {
                  data: (graphView === 'weekly'
                    ? graphData[selectedMetric].slice(0, 7)
                    : graphData[selectedMetric].slice(0, 4)
                  ).map(v => parseFloat(v.toFixed(1))),
                },
              ],
            }}
            width={screenWidth - 80}
            height={220}
            yAxisSuffix={
              selectedMetric === 'weight'
                ? 'kg'
                : selectedMetric === 'bodyFat'
                  ? '%'
                  : ''
            }
            chartConfig={{
              backgroundColor: '#ffffff',
              backgroundGradientFrom: '#ffffff',
              backgroundGradientTo: '#ffffff',
              decimalPlaces: 1,
              color: () => Colors.light.colors.primary[500],
              labelColor: () => Colors.light.colors.primary[700],
              style: { borderRadius: 16 },
              propsForDots: {
                r: '6',
                strokeWidth: '2',
                stroke: Colors.light.colors.primary[500],
              },
              propsForBackgroundLines: {
                stroke: Colors.light.colors.secondary[200],
                strokeWidth: 1,
              },
            }}
            bezier
            style={{ marginVertical: 8, borderRadius: 16 }}
          />
        </View>

        {/* Progress Message */}
        <View className="rounded-xl border border-secondary-200 bg-secondary-50 p-4">
          <Text className="text-center font-body text-body2 text-typography-800">
            You've maintained a healthy {selectedMetric} range this week. Keep
            it up!
          </Text>
        </View>
      </ScrollView>
      <Pressable
        className="button-rounded m-4"
        onPress={() => router.push('/(body)/all-logs')}>
        <Text className="button-rounded-text">View All Logs</Text>
      </Pressable>
    </ThemedSafeAreaView>
  );
}
