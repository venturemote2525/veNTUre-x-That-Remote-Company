import DateSelector from '@/components/DateSelector';
import { Text, ThemedSafeAreaView, View } from '@/components/Themed';
import dayjs from 'dayjs';
import { useState } from 'react';
import { ScrollView, Pressable, Dimensions } from 'react-native';
import { LineChart } from 'react-native-chart-kit';

const screenWidth = Dimensions.get('window').width;

const tempBodyData = [
  { id: 1, type: 'weight', value: 60 },
  { id: 2, type: 'bmi', value: 22 },
  { id: 3, type: 'body fat', value: 18 },
];

// Mock weekly/monthly data for graphs
const graphData = {
  weight: [60, 61, 62, 61.5, 63, 62.8, 62],
  bmi: [22, 21.9, 22.1, 22, 22.2, 22.1, 22],
  bodyFat: [18, 18.2, 18.1, 18.3, 18.5, 18.4, 18],
};

export default function BodyScreen() {
  const [selectedDate, setSelectedDate] = useState(dayjs());
  const [graphView, setGraphView] = useState<'weekly' | 'monthly'>('weekly');

  // labels based on selected view
  const labels = graphView === 'weekly'
    ? ['Mon', 'Tue', 'Wed', 'Thu', 'Fri', 'Sat', 'Sun']
    : ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 'Jul'];

  return (
    <ThemedSafeAreaView edges={['top']} className="flex-1 bg-background-1">
      <ScrollView
        contentContainerStyle={{ paddingHorizontal: 16, paddingBottom: 20 }}
        showsVerticalScrollIndicator={false}
      >
        <DateSelector selectedDate={selectedDate} onDateChange={setSelectedDate} />

        {/* Today's Metrics */}
        <ScrollView
          horizontal
          showsHorizontalScrollIndicator={false}
          className="flex-row gap-4 mt-6 mb-6"
        >
          {tempBodyData.map(item => (
            <View
              key={item.id}
              className="bg-background-0 rounded-2xl w-[120px] h-[60px] items-center justify-center"
            >
              <Text className="font-bodyBold text-body2">{item.type.toUpperCase()}</Text>
              <Text className="font-bodySemiBold text-head2 text-primary-500">
                {item.value}
                {item.type === 'weight' ? ' kg' : item.type === 'bmi' ? '' : '%'}
              </Text>
            </View>
          ))}
        </ScrollView>

        {/*weekly and Monthly */}
        <View className="flex-row justify-center gap-4 mb-6">
          <Pressable
            onPress={() => setGraphView('weekly')}
            className={`px-4 py-2 rounded-full ${
              graphView === 'weekly' ? 'bg-primary-500' : 'bg-gray-300'
            }`}
          >
            <Text className="text-background-0">Weekly</Text>
          </Pressable>
          <Pressable
            onPress={() => setGraphView('monthly')}
            className={`px-4 py-2 rounded-full ${
              graphView === 'monthly' ? 'bg-primary-500' : 'bg-gray-300'
            }`}
          >
            <Text className="text-background-0">Monthly</Text>
          </Pressable>
        </View>

        {/* Graphs */}
        {(['weight', 'bmi', 'bodyFat'] as const).map((metric) => (
  <View className="mb-6" key={metric}>
    <Text className="font-bodyBold text-body1 mb-1">
      {metric === 'weight' ? 'Weight' : metric === 'bmi' ? 'BMI' : 'Body Fat'}
    </Text>
    <LineChart
      data={{
        labels: labels,
        datasets: [{ data: graphData[metric] }], // ✅ metric is now typed correctly
      }}
      width={screenWidth - 32}
      height={220}
      yAxisSuffix={metric === 'weight' ? 'kg' : metric === 'bodyFat' ? '%' : ''}
      chartConfig={{
        backgroundColor: '#fff',
        backgroundGradientFrom: '#fff',
        backgroundGradientTo: '#fff',
        color: () =>
          metric === 'weight' ? '#3b82f6' :
          metric === 'bmi' ? '#10b981' :
          '#f97316',
      }}
      style={{ borderRadius: 16 }}
    />
  </View>
))}

      </ScrollView>
    </ThemedSafeAreaView>
  );
}
