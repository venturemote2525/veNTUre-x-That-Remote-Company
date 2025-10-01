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
import Ionicons from '@expo/vector-icons/Ionicons';
import ScrollChart from '@/components/Body/ScrollChart';
import {
  fetchGroupedManualLogs,
  fetchGroupedScaleLogs,
} from '@/utils/body/api';
import {
  DateGroup,
  MergedLogSummary,
  MetricType,
  ScaleLogSummary,
} from '@/types/database-types';
import { useFocusEffect } from '@react-navigation/native';
import { mergeGroupedLogs, transformScaleLogs } from '@/utils/body/body';

const TABS = ['weight', 'BMI', 'body_fat', 'height'];

export default function BodyScreen() {
  const [screen, setScreen] = useState<(typeof TABS)[number]>('weight');
  const [dateGroup, setDateGroup] = useState<DateGroup>('WEEK');
  const router = useRouter();
  const [cache, setCache] = useState<Record<DateGroup, MergedLogSummary[]>>(
    {} as Record<DateGroup, MergedLogSummary[]>,
  );
  const [data, setData] = useState<MergedLogSummary[] | null>(null);

  // Animation values - same as profile screen
  const fadeAnim = useState(new Animated.Value(0))[0];
  const slideAnim = useState(new Animated.Value(50))[0];
  const scaleAnim = useState(new Animated.Value(0.95))[0];

  // Start animations when component mounts
  useEffect(() => {
    Animated.parallel([
      Animated.timing(fadeAnim, {
        toValue: 1,
        duration: 600,
        useNativeDriver: true,
      }),
      Animated.timing(slideAnim, {
        toValue: 0,
        duration: 500,
        useNativeDriver: true,
      }),
      Animated.timing(scaleAnim, {
        toValue: 1,
        duration: 400,
        useNativeDriver: true,
      }),
    ]).start();
  }, []);

  const overview = useMemo(() => {
    if (!data || data.length === 0) return { current: '-', previous: '-' };

    const sorted = [...data].sort(
      (a, b) => new Date(b.start).getTime() - new Date(a.start).getTime(),
    );

    const getValidValue = (entry: MergedLogSummary, type: typeof screen) => {
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
        case 'height':
          return entry.average_height && entry.average_height !== 0
            ? entry.average_height
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
      case 'height':
        return 'cm';
      default:
        return '';
    }
  }, [screen]);

  useFocusEffect(
    useCallback(() => {
      console.log('Screen focused');

      (async () => {
        try {
          const [scaleResult, manualResult] = await Promise.all([
            fetchGroupedScaleLogs(dateGroup),
            fetchGroupedManualLogs(dateGroup),
          ]);
          const merged = mergeGroupedLogs(scaleResult, manualResult);
          setData(merged);
          setCache(prev => ({ ...prev, [dateGroup]: merged }));
          console.log('Fetched data for', dateGroup, merged);
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

      <Animated.ScrollView 
        style={{ 
          opacity: fadeAnim,
          transform: [{ translateY: slideAnim }]
        }}
        className="flex-1 px-4"
        showsVerticalScrollIndicator={false}
      >
        <View className="items-center justify-center gap-4 py-4">
          <OverviewCard
            dateGroup={dateGroup}
            current={overview.current}
            previous={overview.previous}
            graphLabel={graphLabel}
          />
          
          {/* Duration Selector*/}
          <Animated.View 
            style={{ transform: [{ scale: scaleAnim }] }}
            className="flex-row justify-center gap-4 w-full"
          >
            <Pressable
              onPress={() => setDateGroup('WEEK')}
              className={`flex-1 flex-row items-center justify-center gap-2 rounded-full px-4 py-3 shadow-md ${dateGroup === 'WEEK' ? 'bg-secondary-500' : 'bg-background-0'}`}>
              <Ionicons 
                name="calendar-outline" 
                size={16} 
                color={dateGroup === 'WEEK' ? '#fff' : '#6b7280'} 
              />
              <Text
                className={`font-bodySemiBold ${dateGroup === 'WEEK' ? 'text-background-0' : 'text-primary-100'}`}>
                Weekly
              </Text>
            </Pressable>
            <Pressable
              onPress={() => setDateGroup('MONTH')}
              className={`flex-1 flex-row items-center justify-center gap-2 rounded-full px-4 py-3 shadow-md ${dateGroup === 'MONTH' ? 'bg-secondary-500' : 'bg-background-0'}`}>
              <Ionicons 
                name="calendar" 
                size={16} 
                color={dateGroup === 'MONTH' ? '#fff' : '#6b7280'} 
              />
              <Text
                className={`font-bodySemiBold ${dateGroup === 'MONTH' ? 'text-background-0' : 'text-primary-100'}`}>
                Monthly
              </Text>
            </Pressable>
            <Pressable
              onPress={() => setDateGroup('YEAR')}
              className={`flex-1 flex-row items-center justify-center gap-2 rounded-full px-4 py-3 shadow-md ${dateGroup === 'YEAR' ? 'bg-secondary-500' : 'bg-background-0'}`}>
              <Ionicons 
                name="calendar-sharp" 
                size={16} 
                color={dateGroup === 'YEAR' ? '#fff' : '#6b7280'} 
              />
              <Text
                className={`font-bodySemiBold ${dateGroup === 'YEAR' ? 'text-background-0' : 'text-primary-100'}`}>
                Yearly
              </Text>
            </Pressable>
          </Animated.View>

          {graphData ? (
            <Animated.View 
              style={{ transform: [{ scale: scaleAnim }] }}
              className="rounded-2xl bg-background-0 p-4 shadow-lg w-full"
            >
              <ScrollChart
                graphData={graphData}
                label={graphLabel}
                width={dateGroup === 'MONTH' ? 120 : 70}
                spacing={dateGroup === 'MONTH' ? 80 : 60}
                height={250}
              />
            </Animated.View>
          ) : (
            <Animated.View 
              style={{ 
                height: 250,
                transform: [{ scale: scaleAnim }]
              }}
              className="w-full items-center justify-center rounded-2xl bg-background-0 p-6 shadow-lg"
            >
              <Text className="font-bodySemiBold text-body1 text-primary-500">
                No Data Yet
              </Text>
              <Text className="text-center text-primary-200">
                Log your first entry to start tracking your progress!
              </Text>
            </Animated.View>
          )}
        </View>
      </Animated.ScrollView>

      {/* View All Logs Button */}
      <Animated.View 
        style={{ opacity: fadeAnim }}
        className="px-4 pb-8 pt-4"
      >
        <Pressable
          className="button-rounded m-4 shadow-lg"
          onPress={() => router.push('/(body)/all-logs')}>
          <Text className="button-rounded-text">View All Logs</Text>
        </Pressable>
      </Animated.View>
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
  // Counter animation
  const [displayValue, setDisplayValue] = useState(0);
  const animatedValue = useRef(new Animated.Value(0)).current;

  useEffect(() => {
    if (typeof current === 'number') {
      // Reset to 0 when current changes
      setDisplayValue(0);
      
      // Animate from 0 to current value
      Animated.timing(animatedValue, {
        toValue: current,
        duration: 1500,
        useNativeDriver: false,
      }).start();
      
      // Update display value during animation
      animatedValue.addListener(({ value }) => {
        setDisplayValue(Number(value.toFixed(1)));
      });

      return () => {
        animatedValue.removeAllListeners();
      };
    }
  }, [current]);

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

  return (
    <View className="w-full flex-row gap-4">
      {/* Current with counter animation */}
      <View
        style={{ flex: 1 }}
        className="items-center justify-center gap-2 rounded-2xl bg-background-0 p-4 shadow-md">
        <View className="flex-row items-center gap-1">
          <Ionicons name="today-outline" size={16} color="#6b7280" />
          <Text className="text-primary-300">Current</Text>
        </View>
        <Text className="font-bodySemiBold text-body1 text-secondary-500">
          {typeof current === 'number' ? displayValue : current} {graphLabel}
        </Text>
      </View>

      {/* Trend */}
      <View
        style={{ flex: 1 }}
        className="items-center justify-center gap-2 rounded-2xl bg-background-0 p-4 shadow-md">
        <View className="flex-row items-center gap-1">
          <Ionicons name="trending-up-outline" size={16} color="#6b7280" />
          <Text className="text-primary-300">Trend</Text>
        </View>
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
