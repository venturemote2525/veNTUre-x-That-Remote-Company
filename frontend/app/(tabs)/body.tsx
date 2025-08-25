import { Text, ThemedSafeAreaView, View } from '@/components/Themed';

export default function BodyScreen() {
  return (
    <ThemedSafeAreaView>
      <View className="flex-1 items-center justify-center">
        <Text>Body</Text>
      </View>
    </ThemedSafeAreaView>
  );
}
