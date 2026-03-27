import { FontAwesome, MaterialIcons } from '@expo/vector-icons';
import { useRouter } from 'expo-router';
import {
  Image,
  ScrollView,
  StyleSheet,
  Text,
  TouchableOpacity,
  View,
} from 'react-native';
import { SafeAreaView } from 'react-native-safe-area-context';


// Local doctor images (Assuming these are correctly linked)
import dr1Image from '../assets/images/dr1.jpg';
import dr2Image from '../assets/images/dr2.jpg';
import dr3Image from '../assets/images/dr3.jpg';
import dr4Image from '../assets/images/dr4.jpg';

export default function DoctorsScreen() {
  const router = useRouter();

  const handleGoBack = () => {
    router.back(); // safer than push('/dashboard')
  };

  return (
    <SafeAreaView style={styles.container}>
      {/* Header */}
      <View style={styles.header}>
        <TouchableOpacity onPress={handleGoBack} style={styles.backButton}>
          <MaterialIcons name="arrow-back-ios" size={24} color="#555" />
        </TouchableOpacity>
        <Text style={styles.headerTitle}>Our Specialists</Text>
        <View style={{ width: 24 }} />
      </View>

      <ScrollView contentContainerStyle={styles.content}>
        {/* Doctor 1 */}
        <DoctorCard
          image={dr2Image}
          name="Dr. Mikaela Sherry Lopez"
          title="Optometrist"
          experience="10 years experience"
          focus="Focus: specializes in vision therapy and non-surgical management of strabismus in children and adolescents (18 years and below)."
          hours="M-S, 9 AM - 10 PM"
        />

        {/* Doctor 2 */}
        <DoctorCard
          image={dr1Image}
          name="Dr. Maria Cherry Lopez"
          title="Optometrist"
          experience="20 years experience"
          focus="Focus: comprehensive management of strabismus in adults (19 years and above), including both non-surgical and surgical options."
          hours="M-S, 9 AM - 10 PM"
        />

        {/* Doctor 3 */}
        <DoctorCard
          image={dr3Image}
          name="Dr. Jhon Francis Labis"
          title="Optometrist"
          experience="8 years experience"
          focus="Focus: specializes in pediatric eye care and early detection of vision problems in children."
          hours="M-F, 8 AM - 5 PM"
        />

        {/* Doctor 4 */}
        <DoctorCard
          image={dr4Image}
          name="Dr. Art James Marcial"
          title="Optometrist"
          experience="12 years experience"
          focus="Focus: provides comprehensive eye exams and non-surgical vision correction solutions for adults."
          hours="T-S, 10 AM - 7 PM"
        />
      </ScrollView>
    </SafeAreaView>
  );
}

// Reusable Doctor Card component
function DoctorCard({ image, name, title, experience, focus, hours }) {
  return (
    <View style={styles.card}>
      <View style={styles.profileRow}>
        <Image source={image} style={styles.profilePic} />
        <View style={styles.infoBlock}>
          {/* Name and Title moved closer to the image block */}
          <Text style={styles.doctorName}>{name}</Text>
          <Text style={styles.doctorTitle}>{title}</Text>
          
          <View style={styles.experienceBadge}>
            <FontAwesome
              name="trophy"
              size={14}
              color="#1E3A8A"
              style={{ marginRight: 5 }}
            />
            <Text style={styles.experienceText}>{experience}</Text>
          </View>
        </View>
      </View>
      
      {/* Focus details are separated and bolded slightly */}
      <View style={styles.focusContainer}>
        <Text style={styles.focusText}>{focus}</Text>
      </View>

      {/* Booking/Hours call to action */}
      <TouchableOpacity style={styles.hoursContainer}>
        <MaterialIcons
          name="access-time"
          size={18}
          color="#fff"
          style={{ marginRight: 5 }}
        />
        <Text style={styles.hoursText}>Book Appointment - {hours}</Text>
      </TouchableOpacity>
    </View>
  );
}

const styles = StyleSheet.create({
  // --- Global Styles ---
  container: {
    flex: 1,
    // Corrected background color typo
    backgroundColor: '#C8EAF7', 
  },
  
  // --- Header Styles ---
  header: {
    flexDirection: 'row',
    alignItems: 'center',
    justifyContent: 'space-between',
    paddingHorizontal: 20,
    paddingVertical: 15,
    backgroundColor: '#C8EAF7',
    borderBottomWidth: 1,
    borderBottomColor: '#B0DCE9', // Lighter border
  },
  backButton: {
    padding: 5,
    marginTop: 50,
  },
  headerTitle: {
    fontSize: 22, // Slightly larger
    fontWeight: '700', // Bolder title
    color: '#1E3A8A',
    marginTop: 50,
  },
  
  // --- Content and Scroll View ---
  content: {
    padding: 20,
  },

  // --- Doctor Card Styles ---
  card: {
    width: '100%',
    backgroundColor: '#fff', // Changed to white for better contrast
    borderRadius: 18,
    padding: 20,
    // Enhanced shadow for a lifted effect
    elevation: 8,
    shadowColor: '#1E3A8A',
    shadowOffset: { width: 0, height: 4 },
    shadowOpacity: 0.1,
    shadowRadius: 6,
    marginBottom: 20,
  },
  profileRow: {
    flexDirection: 'row',
    alignItems: 'flex-start',
    marginBottom: 15,
  },
  profilePic: {
    width: 90,
    height: 90,
    borderRadius: 45,
    marginRight: 15,
    borderWidth: 3,
    borderColor: '#77CDE0', // Added a subtle border
  },
  infoBlock: {
    flex: 1,
    justifyContent: 'center',
  },
  
  doctorName: {
    fontSize: 18,
    color: '#1E3A8A', // Use primary dark blue
    fontWeight: 'bold',
    marginBottom: 2,
  },
  doctorTitle: {
    fontSize: 14,
    color: '#555',
    fontWeight: '500',
    marginBottom: 10,
  },

  experienceBadge: {
    flexDirection: 'row',
    backgroundColor: '#FFD700',
    paddingHorizontal: 12,
    paddingVertical: 5,
    borderRadius: 15,
    alignSelf: 'flex-start', // Important for width control
    alignItems: 'center',
  },
  experienceText: {
    fontSize: 13,
    color: '#313131',
    fontWeight: '600',
  },
  
  focusContainer: {
    borderTopWidth: 1,
    borderTopColor: '#eee',
    paddingTop: 15,
    paddingBottom: 5,
  },
  focusText: {
    color: '#313131',
    fontSize: 14,
    fontWeight: '500',
    lineHeight: 20,
  },
  
  hoursContainer: {
    flexDirection: 'row',
    alignItems: 'center',
    justifyContent: 'center',
    backgroundColor: '#77CDE0',
    borderRadius: 25, // More rounded pill shape
    paddingVertical: 12,
    marginTop: 20,
    shadowColor: '#77CDE0',
    shadowOffset: { width: 0, height: 2 },
    shadowOpacity: 0.4,
    shadowRadius: 4,
    elevation: 5,
  },
  hoursText: {
    color: '#fff',
    // Reduced font size slightly for better fitting
    fontSize: 14, 
    fontWeight: '700',
    // Added flexShrink to allow the text to wrap if it's too long
    flexShrink: 1, 
    textAlign: 'center',
  },
});