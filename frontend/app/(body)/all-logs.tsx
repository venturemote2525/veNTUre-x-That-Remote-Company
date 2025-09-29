import { ThemedSafeAreaView, Text, View } from '@/components/Themed';
import DateSelector from '@/components/DateSelector';
import { useEffect, useMemo, useState } from 'react';
import dayjs from 'dayjs';
import Header from '@/components/Header';
import { fetchAllManualLogs, fetchAllScaleLogs } from '@/utils/body/api';
import { ManualLog, MergedLog, ScaleLog } from '@/types/database-types';
import Feather from '@expo/vector-icons/Feather';
import { ScrollView } from 'react-native';

export default function AllLogs() {
  const [selectedDate, setSelectedDate] = useState(dayjs());
  const [scaleData, setScaleData] = useState<ScaleLog[] | null>(null);
  const [manualData, setManualData] = useState<ManualLog[] | null>(null);

  useEffect(() => {
    (async () => {
      try {
        const [scaleResult, manualResult] = await Promise.all([
          fetchAllScaleLogs(),
          fetchAllManualLogs(),
        ]);
        setScaleData(scaleResult);
        setManualData(manualResult);
        console.log(scaleResult)
      } catch (error) {
        console.log('Error fetching all weight logs: ', error);
      }
    })();
  }, [selectedDate]);

  const mergedData = useMemo<MergedLog[]>(() => {
    if (!scaleData || !manualData) return [];

    const normalisedManual: MergedLog[] =
      manualData?.map(log => ({
        id: `manual-${log.id}`,
        user_id: log.user_id,
        weight: log.weight ?? null,
        height: log.height ?? null,
        bmi: null,
        body_fat: null,
        source: 'manual',
        logged_at: log.logged_at ?? log.created_at,
      })) ?? [];

    const normalisedScale: MergedLog[] =
      scaleData?.map(log => ({
        id: `scale-${log.id}`,
        user_id: log.user_id,
        weight: log.weight ?? null,
        height: null,
        bmi: log.bmi ?? null,
        body_fat: log.body_fat ?? null,
        source: 'scale',
        logged_at: log.created_at,
      })) ?? [];

    return [...normalisedManual, ...normalisedScale].sort(
      (a, b) => dayjs(a.logged_at).valueOf() - dayjs(b.logged_at).valueOf(),
    );
  }, [scaleData, manualData]);

  const filteredData = useMemo(() => {
    return mergedData.filter(log =>
      dayjs(log.logged_at).isSame(selectedDate, 'day'),
    );
  }, [mergedData, selectedDate]);

  return (
    <ThemedSafeAreaView>
      <Header title="All Logs" />

      <View className="flex-1 px-4">
        {/* Date Selector */}
        <DateSelector
          selectedDate={selectedDate}
          onDateChange={setSelectedDate}
        />
        <ScrollView className="flex-1" showsVerticalScrollIndicator={false}>
          {filteredData.length > 0 ? (
            filteredData.map(log => {
              return (
                <View
                  key={log.id}
                  className="mb-2 flex-row items-center gap-4 rounded-2xl bg-background-0 p-4 shadow-sm">
                  {/* Time Icon */}
                  <View className="items-center justify-center gap-2">
                    <View className="rounded-full bg-secondary-500 p-3">
                      {dayjs(log.logged_at).hour() >= 6 &&
                      dayjs(log.logged_at).hour() < 18 ? (
                        <Feather name="sun" size={16} color="white" />
                      ) : (
                        <Feather name="moon" size={16} color="white" />
                      )}
                    </View>
                    <Text className="font-bodySemiBold text-secondary-500">
                      {dayjs(log.logged_at).format('HH:mm')}
                    </Text>
                  </View>
                  {/* Separator */}
                  <View
                    className="h-full rounded-full bg-secondary-500"
                    style={{ width: 2 }}
                  />
                  {/* Log Details */}
                  <View
                    className={`flex-1 flex-row items-center px-2 ${log.source === 'manual' ? 'justify-center' : 'justify-between'}`}>
                    {log.weight && (
                      <View className="items-center">
                        <Text className="font-bodySemiBold text-body2 text-primary-500">
                          {log.weight} kg
                        </Text>
                        <Text className="text-body3 text-primary-200">
                          Weight
                        </Text>
                      </View>
                    )}
                    {log.bmi && (
                      <View className="items-center">
                        <Text className="font-bodySemiBold text-body2 text-primary-500">
                          {log.bmi}
                        </Text>
                        <Text className="text-body3 text-primary-200">BMI</Text>
                      </View>
                    )}
                    {log.body_fat && (
                      <View className="items-center">
                        <Text className="font-bodySemiBold text-body2 text-primary-500">
                          {log.body_fat} %
                        </Text>
                        <Text className="text-body3 text-primary-200">
                          Body Fat
                        </Text>
                      </View>
                    )}
                    {log.height && (
                      <View className="items-center">
                        <Text className="font-bodySemiBold text-body2 text-primary-500">
                          {log.height} cm
                        </Text>
                        <Text className="text-body3 text-primary-200">
                          Height
                        </Text>
                      </View>
                    )}
                  </View>
                </View>
              );
            })
          ) : (
            <Text className="mt-4 text-center text-primary-300">
              No logs for this date
            </Text>
          )}
        </ScrollView>
      </View>
    </ThemedSafeAreaView>
  );
}
