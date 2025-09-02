import { Tabs } from 'expo-router';
import TabBar from '@/components/TabBar';
import { FontAwesomeIcon } from '@fortawesome/react-native-fontawesome';
import {
  faHome,
  faUtensils,
  faCamera,
  faChild,
  faUser,
} from '@fortawesome/free-solid-svg-icons';

export default function TabLayout() {
  return (
    <Tabs
      tabBar={props => <TabBar {...props} />}
      screenOptions={{
        headerShown: false,
      }}>
      <Tabs.Screen
        name="home"
        options={{
          tabBarIcon: ({ color }) => (
            <FontAwesomeIcon icon={faHome} size={24} color={color} />
          ),
        }}
      />
      <Tabs.Screen
        name="food"
        options={{
          tabBarIcon: ({ color }) => (
            <FontAwesomeIcon icon={faUtensils} size={24} color={color} />
          ),
        }}
      />
      <Tabs.Screen
        name="logging"
        options={{
          tabBarIcon: ({ color }) => (
            <FontAwesomeIcon icon={faCamera} size={24} color={color} />
          ),
        }}
      />
      <Tabs.Screen
        name="body"
        options={{
          tabBarIcon: ({ color }) => (
            <FontAwesomeIcon icon={faChild} size={24} color={color} />
          ),
        }}
      />
      <Tabs.Screen
        name="profile"
        options={{
          tabBarIcon: ({ color }) => (
            <FontAwesomeIcon icon={faUser} size={24} color={color} />
          ),
        }}
      />
      <Tabs.Screen
        name="capybara"
        options={{
          tabBarIcon: ({ color }) => (
            <FontAwesomeIcon icon={faChild} size={24} color={color} /> // choose an icon
          ),
        }}
      />
    </Tabs>
  );
}
