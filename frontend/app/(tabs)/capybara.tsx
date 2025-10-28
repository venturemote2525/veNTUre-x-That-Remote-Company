import { ThemedSafeAreaView, Text } from '@/components/Themed';
import {
  AnimatedPressable,
  BouncingIcon,
} from '@/components/AnimatedComponents';
import { FontAwesomeIcon } from '@fortawesome/react-native-fontawesome';
import { faOtter } from '@fortawesome/free-solid-svg-icons';
import { Modal, Pressable } from 'react-native';
import { useState } from 'react';
import PetScene from '@/app/(pet)/pet-scene';

export default function CapybaraScreen() {
  const [showScene, setShowScene] = useState<boolean>(false);

  const handleBackgroundPress = () => setShowScene(true);

  return (
    <ThemedSafeAreaView>
      <Pressable
        className="items-center justify-center"
        onPress={handleBackgroundPress}
        style={{ flex: 1 }}>
        <BouncingIcon>
          <FontAwesomeIcon icon={faOtter} size={64} color="#3b82f6" />
        </BouncingIcon>
        <Text className="mt-4 font-bodyBold text-head2 text-primary-500">
          Capybara Tab 🦫
        </Text>
        <Text className="mb-8 text-body1 text-primary-500">
          This is your permanent Capybara tab.
        </Text>

        <AnimatedPressable
          className="rounded-full bg-secondary-500 px-6 py-3"
          onPress={() => console.log('Capybara pressed!')}>
          <Text className="font-bodyBold text-white">Pet the Capybara</Text>
        </AnimatedPressable>
      </Pressable>
      <Modal
        visible={showScene}
        animationType="fade"
        transparent={true}
        onRequestClose={() => setShowScene(false)}>
        <Pressable
          style={{
            flex: 1,
            backgroundColor: 'rgba(0,0,0,0.5)',
            justifyContent: 'center',
            alignItems: 'center',
          }}
          onPress={() => setShowScene(false)}>
          <PetScene />
        </Pressable>
      </Modal>
    </ThemedSafeAreaView>
  );
}
