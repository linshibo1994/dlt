import gsap from 'gsap'

// 元素入场动画
export const animateEnter = (
  element: string | Element,
  options?: gsap.TweenVars
) => {
  return gsap.from(element, {
    opacity: 0,
    y: 30,
    duration: 0.6,
    ease: 'power3.out',
    ...options
  })
}

// 卡片依次入场
export const animateCards = (
  selector: string,
  options?: gsap.TweenVars
) => {
  return gsap.from(selector, {
    opacity: 0,
    y: 50,
    stagger: 0.1,
    duration: 0.8,
    ease: 'power3.out',
    ...options
  })
}

// 号码球弹出动画
export const animateBalls = (
  selector: string,
  options?: gsap.TweenVars
) => {
  return gsap.from(selector, {
    scale: 0,
    rotation: -180,
    stagger: 0.1,
    duration: 0.5,
    ease: 'back.out(1.7)',
    ...options
  })
}

// 数字计数动画
export const animateCount = (
  element: Element,
  endValue: number,
  duration = 1
) => {
  return gsap.to(element, {
    textContent: endValue,
    duration,
    ease: 'power1.out',
    snap: { textContent: 1 },
    modifiers: {
      textContent: (value: string) => Math.round(parseFloat(value)).toString()
    }
  })
}

// 脉冲动画
export const animatePulse = (
  element: string | Element
) => {
  return gsap.to(element, {
    scale: 1.05,
    duration: 0.5,
    ease: 'power2.inOut',
    yoyo: true,
    repeat: -1
  })
}

// 发光动画
export const animateGlow = (
  element: string | Element,
  color = 'rgba(0, 212, 255, 0.6)'
) => {
  return gsap.to(element, {
    boxShadow: `0 0 40px ${color}`,
    duration: 1,
    ease: 'power1.inOut',
    yoyo: true,
    repeat: -1
  })
}

// 停止元素上的所有动画
export const stopAnimation = (element: string | Element) => {
  gsap.killTweensOf(element)
}
