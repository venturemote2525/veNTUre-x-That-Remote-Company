import { View, ScrollView, Dimensions } from 'react-native';
import { Text } from '@/components/Themed';
import { useEffect, useRef } from 'react';
import { LineChart } from 'react-native-gifted-charts';
import { Colors } from '@/constants/Colors';
import { GraphPoint } from '@/types/database-types';

type ScrollChartProps = {
  graphData: GraphPoint[];
  height?: number;
  width?: number;
  spacing?: number;
  sections?: number;
  chartColour?: string;
  label?: string;
};

export default function ScrollChart({
  graphData,
  height = 200,
  width = 50,
  spacing = 50,
  sections = 5,
  chartColour = Colors.light.colors.secondary[500] as string,
  label,
}: ScrollChartProps) {
  const scrollViewRef = useRef<ScrollView>(null);
  // Scroll to end/start on render
  useEffect(() => {
    if (!scrollViewRef.current || !graphData) return;
    if (graphData.length > sections - 1) {
      // Scroll to end
      scrollViewRef.current.scrollToEnd({ animated: true });
    } else {
      // Scroll to start
      scrollViewRef.current.scrollTo({ x: 0, y: 0, animated: true });
    }
  }, [graphData, sections]);

  const screenWidth = Dimensions.get('window').width;
  const minWidth = screenWidth - 80;
  const desiredWidth = graphData.length * spacing;
  const chartWidth = Math.max(minWidth, desiredWidth);

  // 1. Calculate max value from data
  const rawMaxValue = Math.max(...graphData.map(d => d.value), 0);

  // 2. Round max to nearest nice number
  function roundUpMax(value: number) {
    if (value <= 10) return 10;
    if (value <= 25) return 25;
    if (value <= 50) return 50;
    if (value <= 100) return 100;
    // round up to nearest 100
    return Math.ceil(value / 100) * 100;
  }

  // 3. Use dynamically calculated max
  const maxValue = roundUpMax(rawMaxValue);

  // 4. Calculate height of each Y-axis section
  const sectionHeight = height / sections;

  // 5. Generate Y-axis labels
  const sectionLabels = Array.from({ length: sections + 1 }, (_, i) =>
    ((sections - i) * (maxValue / sections)).toFixed(0),
  );

  return (
    <View className="w-full flex-row">
      {/* Fixed Y-axis */}
      <View className="px-2">
        {sectionLabels.map((label, i) => (
          <Text key={i} style={{ color: '#b0aeae', height: sectionHeight }}>
            {label}
          </Text>
        ))}
      </View>

      {/* Scrollable Chart */}
      <ScrollView
        horizontal
        showsHorizontalScrollIndicator={false}
        ref={scrollViewRef}>
        <LineChart
          data={graphData}
          dataPointsColor={chartColour}
          dataPointsRadius={3}
          width={chartWidth}
          height={height}
          thickness={3}
          color={chartColour}
          noOfSections={sections}
          isAnimated
          hideDataPoints={false}
          yAxisColor="transparent"
          xAxisColor="#5d5d5d"
          rulesColor="#5d5d5d"
          xAxisLabelTextStyle={{
            color: '#b0aeae',
            fontFamily: 'Fredoka-Medium',
            height: 24,
            width: width,
          }}
          spacing={spacing}
          maxValue={maxValue}
          hideYAxisText={true}
          pointerConfig={{
            pointerStripUptoDataPoint: true,
            pointerStripColor: '#5d5d5d',
            pointerStripWidth: 2,
            strokeDashArray: [2, 5],
            pointerColor: '#5d5d5d',
            radius: 4,
            pointerLabelWidth: 90,
            shiftPointerLabelY: -20,
            shiftPointerLabelX: -10,
            pointerLabelComponent: (
              items: { value: number; label?: string }[],
            ) => {
              const item = items[0]; // current data point
              return (
                <View
                  style={{ backgroundColor: chartColour }}
                  className="rounded-2xl p-3">
                  <Text style={{ fontSize: 12 }} className="text-background-0 font-bodyBold">
                    {item.label ?? 'Value'}
                  </Text>
                  <Text className="text-background-0">
                    {item.value} {label}
                  </Text>
                </View>
              );
            },
          }}
        />
      </ScrollView>
    </View>
  );
}
