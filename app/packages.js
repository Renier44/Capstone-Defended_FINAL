import React from "react";
import { View, Text, StyleSheet, ScrollView,  TouchableOpacity } from "react-native";
import { MaterialIcons } from '@expo/vector-icons';
import { useRouter } from 'expo-router';
import { SafeAreaView } from 'react-native-safe-area-context';


export default function PackagesScreen() {
  const router = useRouter();

  const handleGoBack = () => {
    router.back(); 
  };
  
  const packages = [
    {
      name: "BASIC",
      price: "P500",
      services: [
        "REFRACTION",
        "INTERNAL EYE SCREEN",
        "EXTERNAL EYE SCREEN",
        "MEDICAL CERTIFICATE",
      ],
    },
    {
      name: "PREMIUM",
      price: "P750",
      services: [
        "REFRACTION",
        "INTERNAL EYE SCREEN",
        "EXTERNAL EYE SCREEN",
        "EYE PRESSURE TEST (GLAUCOMA SCREENING)",
        "MEDICAL CERTIFICATE",
      ],
    },
    {
      name: "STUDENT/WORK",
      price: "P650",
      services: [
        "REFRACTION",
        "INTERNAL EYE SCREEN",
        "EXTERNAL EYE SCREEN",
        "COLOR VISION TESTING",
        "MEDICAL CERTIFICATE",
      ],
    },
    {
      name: "DRY EYE TEST",
      price: "P550",
      services: [
        "SLIT LAMP",
        "INTERNAL EYE SCREEN",
        "EXTERNAL EYE SCREEN",
        "SCHIRMER TEST",
        "FLUORESCEIN STAINING",
        "MEIBOGRAPHY",
      ],
    },
    {
      name: "CHILDREN",
      price: "P1,250",
      subtitle: "by appointment basis only",
      services: [
        "REFRACTION",
        "INTERNAL EYE SCREEN",
        "EXTERNAL EYE SCREEN",
        "COLOR VISION TESTING",
        "MYOPIA MANAGEMENT",
        "STRABISMUS/LAZY EYE SCREENING",
        "MEDICAL CERTIFICATE",
      ],
    },
  ];

  return (
    <SafeAreaView style={styles.container}>
      
      {/* Header */}
      <View style={styles.header}>
        <TouchableOpacity onPress={handleGoBack} style={styles.backButton}>
          <MaterialIcons name="arrow-back-ios" size={24} color="#555" />
        </TouchableOpacity>
        <Text style={styles.headerTitle}>Service Packages</Text>
        <View style={{ width: 24 }} />
      </View>

      {/* Scroll Content */}
      <ScrollView contentContainerStyle={styles.content}>
        {packages.map((pkg, index) => (
          <View key={index} style={styles.packageBox}>
            
            <View style={styles.packageHeader}>
              <Text style={styles.packageName}>{pkg.name}</Text>
              <View style={styles.priceBadge}>
                <Text style={styles.packagePrice}>{pkg.price}</Text>
              </View>
            </View>

            {pkg.subtitle && (
              <Text style={styles.packageSubtitle}>{pkg.subtitle}</Text>
            )}

            <View style={styles.servicesList}>
              {pkg.services.map((service, i) => (
                <View key={i} style={styles.serviceItem}>
                  <MaterialIcons
                    name="check-circle"
                    size={16}
                    color="#1E3A8A"
                    style={styles.serviceIcon}
                  />
                  <Text style={styles.packageService}>{service}</Text>
                </View>
              ))}
            </View>

          </View>
        ))}
      </ScrollView>

      {/* Background Floating Logo - FIXED */}
      <View style={styles.backgroundLogoContainer}>
        <Text style={styles.backgroundLogoText}>Eye Care</Text>
      </View>

    </SafeAreaView>
  );
}

const styles = StyleSheet.create({
  container: {
    flex: 1,
    backgroundColor: '#C8EAF7', 
  },

  header: {
    flexDirection: 'row',
    alignItems: 'center',
    justifyContent: 'space-between',
    paddingHorizontal: 20,
    paddingVertical: 15,
    backgroundColor: '#C8EAF7',
    borderBottomWidth: 1,
    borderBottomColor: '#B0DCE9',
  },
  backButton: {
    padding: 5,
    marginTop: 50,
  },
  headerTitle: {
    fontSize: 22,
    fontWeight: '700',
    color: '#1E3A8A',
    marginTop: 50,
  },

  content: {
    padding: 20,
    paddingBottom: 120,
  },

  packageBox: {
    backgroundColor: '#FFFFFF',
    borderRadius: 18,
    padding: 20,
    elevation: 8,
    shadowColor: '#1E3A8A',
    shadowOffset: { width: 0, height: 4 },
    shadowOpacity: 0.1,
    shadowRadius: 6,
    marginBottom: 20,

    borderLeftWidth: 8,
    borderLeftColor: '#77CDE0',
    paddingLeft: 12,
  },

  packageHeader: {
    flexDirection: "row",
    justifyContent: "space-between",
    alignItems: 'center',
    marginBottom: 10,
    paddingBottom: 10,
    borderBottomWidth: 1,
    borderBottomColor: '#E0E0E0',
  },

  packageName: {
    fontSize: 20,
    fontWeight: "bold",
    color: '#1E3A8A',
    flexShrink: 1,
    marginRight: 10,
  },

  priceBadge: {
    backgroundColor: '#FFD700',
    paddingHorizontal: 15,
    paddingVertical: 8,
    borderRadius: 20,
    minWidth: 80,
    alignItems: 'center',
  },

  packagePrice: {
    fontSize: 18,
    fontWeight: "800",
    color: '#1E3A8A',
  },

  packageSubtitle: {
    fontSize: 13,
    fontStyle: "italic",
    color: '#777',
    marginBottom: 10,
  },

  servicesList: {
    marginTop: 5,
  },

  serviceItem: {
    flexDirection: 'row',
    alignItems: 'center',
    marginVertical: 4,
  },

  packageService: {
    fontSize: 14,
    color: '#444',
    marginLeft: 8,
    fontWeight: '500',
  },

  serviceIcon: {
    minWidth: 16,
  },

  backgroundLogoContainer: {
    position: "absolute",
    right: 30,
    bottom: 30,
    width: 180,
    height: 180,
    justifyContent: 'center',
    alignItems: 'center',
  },

  backgroundLogoText: {
    fontSize: 40,
    fontWeight: '900',
    color: '#1E3A8A',
    opacity: 0.08,
    textAlign: 'center',
    transform: [{ rotate: '-15deg' }],
  },
});
