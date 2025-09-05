import React, { useRef, useState, useEffect } from 'react';
import { Animated, StyleSheet, Keyboard, TouchableOpacity } from 'react-native';
import { View, TextInput } from '@/components/Themed';

const ITEM_HEIGHT = 50;
const VISIBLE_ITEMS = 3;

export default function CustomWheelPicker({
  data,
  onSelect,
  initialIndex = 0,
  selectedIndex: externalSelectedIndex,
  minValue,
  maxValue,
  zeroPad,
  textSize = 18,
}: {
  data: number[];
  onSelect: (value: number) => void;
  initialIndex?: number;
  selectedIndex?: number;
  minValue?: number;
  maxValue?: number;
  zeroPad?: boolean;
  textSize?: number;
}) {
  const [selectedIndex, setSelectedIndexInternal] = useState(initialIndex);
  const [inputValue, setInputValue] = useState(
    data[initialIndex]?.toString() ?? '',
  );
  const [isEditing, setIsEditing] = useState(false);

  const scrollY = useRef(
    new Animated.Value(initialIndex * ITEM_HEIGHT),
  ).current;
  const flatListRef = useRef<Animated.FlatList<number>>(null);

  useEffect(() => {
    const newIndex = externalSelectedIndex ?? selectedIndex;
    if (data[newIndex] !== undefined && flatListRef.current) {
      flatListRef.current.scrollToOffset({
        offset: newIndex * ITEM_HEIGHT,
        animated: true,
      });
      setInputValue(data[newIndex].toString());
    }
  }, [externalSelectedIndex, selectedIndex, data]);

  const updateSelection = (index: number) => {
    if (data[index] !== undefined) {
      setSelectedIndexInternal(index);
      setInputValue(data[index].toString());
      onSelect(data[index]);
    }
  };

  const onInputSubmit = () => {
    setIsEditing(false);
    if (inputValue === '') {
      setInputValue(data[selectedIndex].toString());
      return;
    }

    // Clamp to min/max
    let numericValue = Number(inputValue);
    if (minValue !== undefined && numericValue < minValue) {
      numericValue = minValue;
    }
    if (maxValue !== undefined && numericValue > maxValue) {
      numericValue = maxValue;
    }

    const nearestIndex = data.reduce((prevIndex, currValue, currIndex) => {
      const prevDiff = Math.abs(data[prevIndex] - numericValue);
      const currDiff = Math.abs(currValue - numericValue);
      return currDiff < prevDiff ? currIndex : prevIndex;
    }, 0);

    // Update selection and scroll the wheel
    updateSelection(nearestIndex);

    Keyboard.dismiss();
  };

  const onScrollEnd = (event: any) => {
    const offsetY = event.nativeEvent.contentOffset.y;
    const index = Math.round(offsetY / ITEM_HEIGHT);
    updateSelection(index);
  };

  const effectiveSelectedIndex = externalSelectedIndex ?? selectedIndex;

  return (
    <View
      style={{
        height: ITEM_HEIGHT * VISIBLE_ITEMS,
        alignItems: 'center',
      }}>
      <Animated.FlatList
        data={data}
        ref={flatListRef}
        keyExtractor={item => item.toString()}
        showsVerticalScrollIndicator={false}
        snapToInterval={ITEM_HEIGHT}
        snapToAlignment="center"
        decelerationRate="fast"
        scrollEventThrottle={16}
        onMomentumScrollEnd={onScrollEnd}
        onScroll={Animated.event(
          [{ nativeEvent: { contentOffset: { y: scrollY } } }],
          { useNativeDriver: true },
        )}
        removeClippedSubviews={false}
        getItemLayout={(_, index) => ({
          length: ITEM_HEIGHT,
          offset: ITEM_HEIGHT * index,
          index,
        })}
        keyboardShouldPersistTaps="handled"
        style={{ flexGrow: 0 }}
        contentContainerStyle={{ paddingVertical: ITEM_HEIGHT }}
        renderItem={({ item, index }) => {
          const offset = ITEM_HEIGHT * index;
          const inputRange = [
            offset - ITEM_HEIGHT,
            offset,
            offset + ITEM_HEIGHT,
          ];

          const opacity = scrollY.interpolate({
            inputRange,
            outputRange: [0.3, 1, 0.3],
            extrapolate: 'clamp',
          });

          const scale = scrollY.interpolate({
            inputRange,
            outputRange: [0.9, 1, 0.9],
            extrapolate: 'clamp',
          });

          const isSelected = index === effectiveSelectedIndex;

          if (isSelected && isEditing) {
            return (
              <View style={styles.item}>
                <TextInput
                  className="font-bodySemiBold text-secondary-500"
                  style={[
                    {
                      textAlign: 'center',
                      fontSize: textSize,
                      height: textSize * 1.8,
                    },
                  ]}
                  autoFocus
                  keyboardType="numeric"
                  value={inputValue}
                  onBlur={onInputSubmit}
                  onChangeText={setInputValue}
                />
              </View>
            );
          }

          return (
            <TouchableOpacity
              onPress={() => {
                if (isSelected) {
                  setIsEditing(true);
                } else {
                  updateSelection(index);
                }
              }}
              style={styles.item}
              activeOpacity={0.7}>
              <Animated.Text
                className="text-primary-500"
                style={[
                  {
                    opacity,
                    transform: [{ scale }],
                    fontFamily: 'Fredoka-Regular',
                    fontSize: textSize,
                  },
                ]}>
                {zeroPad ? item.toFixed(1).padStart(2, '0') : item.toFixed(1)}
              </Animated.Text>
            </TouchableOpacity>
          );
        }}
      />
    </View>
  );
}

const styles = StyleSheet.create({
  item: {
    height: ITEM_HEIGHT,
    justifyContent: 'center',
    alignItems: 'center',
    backgroundColor: 'transparent',
  },
  itemText: {
    color: '#ebedeb',
  },
});
