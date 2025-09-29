import { Text, ThemedSafeAreaView, View } from '@/components/Themed';
import { memo, useCallback, useEffect, useMemo, useRef, useState } from 'react';
import { ScrollView, Dimensions, Pressable, Animated } from 'react-native';
import { LineChart } from 'react-native-chart-kit';
import { AnimatedPressable } from '@/components/AnimatedComponents';
import { Colors } from '@/constants/Colors';
import TabToggle from '@/components/TabToggle';
import { useRouter } from 'expo-router';
import MaterialIcons from '@expo/vector-icons/MaterialIcons';
import FontAwesome6 from '@expo/vector-icons/FontAwesome6';
import ScrollChart from '@/components/Body/ScrollChart';
import { fetchWeightLogs } from '@/utils/body/api';
import { DateGroup, MetricType, ScaleLogSummary } from '@/types/database-types';
import { useFocusEffect } from '@react-navigation/native';
import { transformScaleLogs } from '@/utils/body/body';

const TABS = ['weight', 'BMI', 'body_fat'];

export default function BodyScreen() {
  const [screen, setScreen] = useState<(typeof TABS)[number]>('weight');
  const [dateGroup, setDateGroup] = useState<DateGroup>('WEEK');
  const router = useRouter();
  const [cache, setCache] = useState<Record<DateGroup, ScaleLogSummary[]>>(
    {} as Record<DateGroup, ScaleLogSummary[]>,
  );
  const [data, setData] = useState<ScaleLogSummary[] | null>(null);

  const overview = useMemo(() => {
    if (!data || data.length === 0) return { current: '-', previous: '-' };

    const sorted = [...data].sort(
      (a, b) => new Date(b.start).getTime() - new Date(a.start).getTime(),
    );

    const getValidValue = (entry: ScaleLogSummary, type: typeof screen) => {
      switch (type) {
        case 'weight':
          return entry.average_weight && entry.average_weight !== 0
            ? entry.average_weight
            : null;
        case 'BMI':
          return entry.average_bmi && entry.average_bmi !== 0
            ? entry.average_bmi
            : null;
        case 'body_fat':
          return entry.average_bodyfat && entry.average_bodyfat !== 0
            ? entry.average_bodyfat
            : null;
      }
    };

    // Find most recent valid current value
    const currentEntry = sorted.find(
      entry => getValidValue(entry, screen) !== null,
    );
    const current = currentEntry ? getValidValue(currentEntry, screen)! : '-';

    // Find previous valid value
    const previousIndex = currentEntry ? sorted.indexOf(currentEntry) + 1 : 0;
    const previousEntry = sorted
      .slice(previousIndex)
      .find(entry => getValidValue(entry, screen) !== null);
    const previous = previousEntry
      ? getValidValue(previousEntry, screen)!
      : '-';

    return { current, previous };
  }, [data, screen]);

  const graphLabel = useMemo(() => {
    switch (screen) {
      case 'weight':
        return 'kg';
      case 'BMI':
        return '';
      case 'body_fat':
        return '%';
      default:
        return '';
    }
  }, [screen]);

  useFocusEffect(
    useCallback(() => {
      console.log('Screen focused');

      (async () => {
        try {
          const result = await fetchWeightLogs(dateGroup);
          setData(result);
          setCache(prev => ({ ...prev, [dateGroup]: result }));
          console.log('Fetched data for', dateGroup, result);
        } catch (error) {
          console.error('Error fetching scale logs:', error);
        }
      })();
    }, [dateGroup]),
  );

  const graphData = useMemo(() => {
    if (!data) return null;
    return transformScaleLogs(data, dateGroup, screen as MetricType);
  }, [data, dateGroup, screen]);

  return (
    <ThemedSafeAreaView edges={['top']} className="flex-1 bg-background-0">
      {/* Tab Selector */}
      <TabToggle tabs={TABS} selectedTab={screen} onTabChange={setScreen} />

      <View className="flex-1 gap-4 p-4">
        <OverviewCard
          dateGroup={dateGroup}
          current={overview.current}
          previous={overview.previous}
          graphLabel={graphLabel}
        />
        {/* Duration Selector*/}
        <View className="flex-row justify-center gap-4">
          <Pressable
            onPress={() => setDateGroup('WEEK')}
            className={`flex-1 items-center rounded-full px-8 py-3 ${dateGroup === 'WEEK' ? 'bg-secondary-500' : 'bg-background-0'}`}>
            <Text
              className={`font-bodySemiBold ${dateGroup === 'WEEK' ? 'text-background-0' : 'text-primary-100'}`}>
              Weekly
            </Text>
          </Pressable>
          <Pressable
            onPress={() => setDateGroup('MONTH')}
            className={`flex-1 items-center rounded-full px-8 py-3 ${dateGroup === 'MONTH' ? 'bg-secondary-500' : 'bg-background-0'}`}>
            <Text
              className={`font-bodySemiBold ${dateGroup === 'MONTH' ? 'text-background-0' : 'text-primary-100'}`}>
              Monthly
            </Text>
          </Pressable>
        </View>
        <View className="rounded-2xl bg-background-0 p-4">
          {graphData ? (
            <ScrollChart
              graphData={graphData}
              label={graphLabel}
              width={dateGroup === 'MONTH' ? 120 : 70}
              spacing={dateGroup === 'MONTH' ? 80 : 60}
            />
          ) : (
            <View>
              <Text>No Data</Text>
            </View>
          )}
        </View>
      </View>
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

const OverviewCardBase = ({
  dateGroup,
  current,
  previous,
  graphLabel,
}: {
  dateGroup: DateGroup;
  current: number | string;
  previous: number | string;
  graphLabel: string;
}) => {
  // Trend Calculation
  let difference: number | null = null;
  if (typeof current === 'number' && typeof previous === 'number') {
    difference = current - previous;
  }
  const trendIcon =
    difference === null ? null : difference > 0 ? (
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
  const trendText = difference === null ? '-' : Math.abs(difference).toFixed(1);
  // Animate flex values
  const flexAnim = useRef(new Animated.Value(1)).current; // start at flex-1
  useEffect(() => {
    Animated.timing(flexAnim, {
      toValue: dateGroup === 'WEEK' ? 1 : 2, // Animate flex value
      duration: 300,
      useNativeDriver: false,
    }).start();
  }, [dateGroup, flexAnim]);

  return (
    <View className="w-full flex-row gap-4">
      {/* Current */}
      <View
        style={{ flex: 1 }}
        className="items-center justify-center gap-2 rounded-2xl bg-background-0 p-4">
        <Text className="text-primary-300">Current</Text>
        <Text className="font-bodySemiBold text-body1 text-secondary-500">
          {current} {graphLabel}
        </Text>
      </View>

      {/* Trend */}
      <View
        style={{ flex: 2 }}
        className="items-center justify-center gap-2 rounded-2xl bg-background-0 p-4">
        <Text className="text-primary-300">Trend</Text>
        <View className="flex-row items-center gap-2">
          {trendIcon}
          <Text className="font-bodySemiBold text-body1 text-secondary-500">
            {trendText}
          </Text>
        </View>
      </View>
    </View>
  );
};

export const OverviewCard = memo(OverviewCardBase);
