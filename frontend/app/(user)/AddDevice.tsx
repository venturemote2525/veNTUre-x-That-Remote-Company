import Header from '@/components/Header';
import { Text, ThemedSafeAreaView, View } from '@/components/Themed';

export default function AddDevice() {
  return (
    <ThemedSafeAreaView>
      <Header title="Add Device" />
      <View className="flex-1 items-center justify-center">
        <Text>Instructions on how to add device here</Text>
      </View>
    </ThemedSafeAreaView>
  );
}
