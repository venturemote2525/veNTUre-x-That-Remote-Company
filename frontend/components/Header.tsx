
import React from 'react';
import { useNavigation } from 'expo-router';
import { LinearGradient } from 'expo-linear-gradient';
import { View, Text } from '@/components/Themed';
import { ChevronLeftIcon, Icon } from '@/components/ui/icon';
import { Colors, Shadows } from '@/constants/Colors';
import { AnimatedPressable, useSlideIn } from '@/components/AnimatedComponents';

type HeaderProps = {
  title?: string;
  onBackPress?: () => void;
  transparent?: boolean;
};

export default function Header({ title, onBackPress, transparent = false }: HeaderProps) {
  const navigation = useNavigation();
  const headerSlide = useSlideIn('down', 0);

  const handleBackPress = () => {
    if (onBackPress) onBackPress();
    else navigation.goBack();
  };

  if (transparent) {
    return (
      <View style={{ 
        flexDirection: 'row', 
        alignItems: 'center', 
        paddingHorizontal: 8, 
        paddingVertical: 12,
        backgroundColor: 'transparent'
      }}>
        <AnimatedPressable onPress={handleBackPress}>
          <View style={{
            width: 40,
            height: 40,
            borderRadius: 20,
            backgroundColor: 'rgba(255, 255, 255, 0.2)',
            justifyContent: 'center',
            alignItems: 'center',
            ...Shadows.small,
          }}>
            <Icon as={ChevronLeftIcon} size="lg" style={{ color: 'white' }} />
          </View>
        </AnimatedPressable>
        
        {title && (
          <Text style={{
            marginLeft: 12,
            fontSize: 18,
            fontWeight: '600',
            color: 'white',
            textShadowColor: 'rgba(0, 0, 0, 0.3)',
            textShadowOffset: { width: 1, height: 1 },
            textShadowRadius: 3,
          }}>
            {title}
          </Text>
        )}
      </View>
    );
  }

  return (
    <View style={{ 
      flexDirection: 'row', 
      alignItems: 'center', 
      paddingHorizontal: 8, 
      paddingVertical: 12 
    }}>
      <AnimatedPressable onPress={handleBackPress}>
        <LinearGradient
          colors={Colors.light.gradients.secondary}
          style={{
            width: 40,
            height: 40,
            borderRadius: 20,
            justifyContent: 'center',
            alignItems: 'center',
            ...Shadows.small,
          }}
        >
          <Icon as={ChevronLeftIcon} size="lg" style={{ color: 'white' }} />
        </LinearGradient>
      </AnimatedPressable>
      
      {title && (
        <Text style={{
          marginLeft: 12,
          fontSize: 18,
          fontWeight: '600',
          color: Colors.light.colors.primary[600],
        }}>
          {title}
        </Text>
      )}
    </View>
  );
}